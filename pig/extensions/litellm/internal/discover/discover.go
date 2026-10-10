// Package discover ports src/discover.ts: model discovery against a LiteLLM proxy.
//
// Ports src/discover.ts
package discover

import (
	"context"
	"fmt"
	"net/url"
	"slices"
	"strings"
	"sync"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/catalog"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/modelgroups"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/proxyversion"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

const healthDetailConcurrency = 8

// orderedGroups is a Map keyed by route that remembers insertion order.
type orderedGroups[V any] struct {
	routes []string
	rows   map[string][]V
}

func (g *orderedGroups[V]) add(route string, row V) {
	if g.rows == nil {
		g.rows = map[string][]V{}
	}
	if _, ok := g.rows[route]; !ok {
		g.routes = append(g.routes, route)
	}
	g.rows[route] = append(g.rows[route], row)
}

// looseRows decodes /v1/models and /health leniently: the TypeScript validates each field with
// wireString, so a mistyped row or field is skipped instead of failing the response.
type looseRows struct {
	Data             []any `json:"data"`
	HealthyEndpoints []any `json:"healthy_endpoints"`
}

func looseString(row any, key string) string {
	fields, _ := row.(map[string]any)
	value, _ := fields[key].(string)
	return value
}

func modelsListEntries(rows []any) []types.ModelsListEntry {
	entries := make([]types.ModelsListEntry, len(rows))
	for i, row := range rows {
		entries[i] = types.ModelsListEntry{ID: looseString(row, "id"), OwnedBy: looseString(row, "owned_by")}
	}
	return entries
}

func isMissingOrDenied(status int) bool {
	return status == 401 || status == 403 || status == 404
}

// DiscoverModels is discoverModels.
func DiscoverModels(ctx context.Context, baseURL, apiKey string, options Options) (*types.DiscoveryResult, error) {
	base, err := protocols.NormalizeBaseURL(baseURL, options.AllowInsecureHTTP)
	if err != nil {
		return nil, err
	}
	options.progress("Querying /model/info endpoint...")
	infoResult, err := FetchJSON[types.ModelInfoResponse](ctx, base+"/model/info", apiKey, options)
	if err != nil {
		return nil, err
	}
	// An empty /model/info list is treated like an unavailable endpoint: some proxies answer 200
	// with no rows for keys that can still list models through /v1/models.
	if infoResult.OK && len(infoResult.Data.Data) > 0 {
		return discoverFromModelInfo(ctx, base, apiKey, options, infoResult.Data)
	}
	if !infoResult.OK && !isMissingOrDenied(infoResult.Status) {
		return nil, fmt.Errorf("/model/info returned %d", infoResult.Status)
	}
	options.progress("/model/info unavailable or empty, trying /v1/models...")
	listResult, err := FetchJSON[looseRows](ctx, base+"/v1/models", apiKey, options)
	if err != nil {
		return nil, err
	}
	if !listResult.OK {
		if isMissingOrDenied(listResult.Status) {
			options.progress("/v1/models unavailable, falling back to /health endpoint...")
			models, err := discoverFromHealth(ctx, base, apiKey, options)
			if err != nil {
				return nil, err
			}
			if len(models) > 0 {
				return &types.DiscoveryResult{Source: types.SourceHealth, Models: deduplicateModels(models)}, nil
			}
		}
		return nil, fmt.Errorf("/v1/models returned %d", listResult.Status)
	}
	models := []types.DiscoveredModel{}
	for _, entry := range modelsListEntries(listResult.Data.Data) {
		if model := mapFromModelsList(entry); model != nil {
			models = append(models, *model)
		}
	}
	return &types.DiscoveryResult{Source: types.SourceModelsList, Models: deduplicateModels(models)}, nil
}

type reducedGroup struct {
	route              string
	model              *types.DiscoveredModel
	deploymentFamilies []modelgroups.FamilyEvidence
}

func discoverFromModelInfo(
	ctx context.Context,
	base, apiKey string,
	options Options,
	info types.ModelInfoResponse,
) (*types.DiscoveryResult, error) {
	var groups orderedGroups[types.ModelInfoEntry]
	for _, entry := range info.Data {
		// A route without a readable public name cannot be grouped or addressed.
		if entry.ModelName == "" {
			continue
		}
		groups.add(entry.ModelName, entry)
	}
	publicCatalog, err := loadDiscoveryPublicCatalog(ctx, options)
	if err != nil {
		return nil, err
	}
	// A wildcard route publishes nothing until `/v1/models` expands it, so only exact routes can call
	// for the proxy version this early.
	proxyVersionRead := false
	readProxyVersion := func() *proxyversion.Version {
		proxyVersionRead = true
		return ProbeProxyVersion(ctx, base, apiKey, options)
	}
	var proxyVersion *proxyversion.Version
	for _, route := range groups.routes {
		if !strings.Contains(route, "*") && publishesVersionGatedTransport(groups.rows[route], publicCatalog) {
			proxyVersion = readProxyVersion()
			break
		}
	}
	ambiguousRoutes, conflictingFamilyRoutes, withheldRepairRoutes := &routeCollector{}, &routeCollector{}, &routeCollector{}
	var incompatibleModeRoutes []string
	defaultedContextRoutes := map[string]bool{}
	reducedGroups := make([]reducedGroup, 0, len(groups.routes))
	for _, route := range groups.routes {
		group := groups.rows[route]
		if modelgroups.HasMixedIncompatibleDeploymentModes(group) {
			incompatibleModeRoutes = append(incompatibleModeRoutes, route)
		}
		families := make([]modelgroups.FamilyEvidence, len(group))
		for i, entry := range group {
			families[i] = deploymentFamily(entry)
		}
		reducedGroups = append(reducedGroups, reducedGroup{
			route: route,
			model: mapFromModelInfoGroup(group, publicCatalog, &groupOptions{
				ambiguousRoutes:         ambiguousRoutes,
				conflictingFamilyRoutes: conflictingFamilyRoutes,
				withheldRepairRoutes:    withheldRepairRoutes,
				defaultedContextRoutes:  defaultedContextRoutes,
				proxyVersion:            proxyVersion,
			}),
			deploymentFamilies: families,
		})
	}
	models := []types.DiscoveredModel{}
	for _, group := range reducedGroups {
		if group.model != nil {
			models = append(models, *group.model)
		}
	}
	reportIncompatibleDeploymentModes(incompatibleModeRoutes, options)
	reportAmbiguousCatalogAuthority(ambiguousRoutes.list(), options)
	reportConflictingFamilyEvidence(conflictingFamilyRoutes.list(), options)
	reportWithheldToolRepair(withheldRepairRoutes.list(), options)

	// LiteLLM's /model/info does NOT expand wildcard model_name entries (e.g. "lemonade/*" backed by
	// model: openai/* + check_provider_endpoint: true) — it returns the literal wildcard only. The
	// discovered ids live in /v1/models instead. When /model/info contains any wildcard id, also query
	// /v1/models and merge the expanded (non-wildcard) entries in, dropping the raw wildcard row so it
	// doesn't surface as a phantom model choice.
	// Ref: docs.litellm.ai/docs/proxy/model_discovery
	var wildcardRoutes []reducedGroup
	for _, group := range reducedGroups {
		if strings.Contains(group.route, "*") {
			wildcardRoutes = append(wildcardRoutes, group)
		}
	}
	if len(wildcardRoutes) > 0 {
		var wildcards []wildcardSource
		var wildcardRows []types.ModelInfoEntry
		publishedWildcardIDs := map[string]bool{}
		for _, group := range wildcardRoutes {
			if group.model != nil {
				wildcards = append(wildcards, wildcardSource{model: *group.model, deploymentFamilies: group.deploymentFamilies})
				publishedWildcardIDs[group.model.ID] = true
			}
			wildcardRows = append(wildcardRows, groups.rows[group.route]...)
		}
		// Exact exclusions are bounded to the same public id: `/v1/models` lacks deployment identity,
		// so a differently named id for that deployment is unknowable.
		droppedExactIDs := map[string]bool{}
		var droppedWildcards []string
		for _, group := range reducedGroups {
			if group.model != nil {
				continue
			}
			if strings.Contains(group.route, "*") {
				droppedWildcards = append(droppedWildcards, group.route)
			} else {
				droppedExactIDs[group.route] = true
			}
		}
		// A wildcard row is not addressable. Remove it before expansion so a failed `/v1/models` request
		// cannot leak the literal wildcard into the selector.
		models = slices.DeleteFunc(models, func(model types.DiscoveredModel) bool { return strings.Contains(model.ID, "*") })
		options.progress("/model/info has wildcard entries, expanding via /v1/models...")
		listResult, err := FetchJSON[looseRows](ctx, base+"/v1/models", apiKey, options)
		if err != nil {
			return nil, err
		}
		if listResult.OK && len(wildcards) > 0 {
			seen := map[string]bool{}
			for _, model := range models {
				seen[model.ID] = true
			}
			var candidates []types.ModelsListEntry
			for _, entry := range modelsListEntries(listResult.Data.Data) {
				id := entry.ID
				if !seen[id] && !droppedExactIDs[id] &&
					!slices.ContainsFunc(droppedWildcards, func(route string) bool { return WildcardMatches(route, id) }) {
					candidates = append(candidates, entry)
				}
			}
			sources := wildcards
			expansionAwaitsProxyVersion := slices.ContainsFunc(candidates, func(entry types.ModelsListEntry) bool {
				if entry.ID == "" || strings.Contains(entry.ID, "*") {
					return false
				}
				parents := selectedWildcardRows(entry.ID, wildcardRows, publishedWildcardIDs)
				return parents != nil && publishesVersionGatedTransport(parents, publicCatalog)
			})
			if !proxyVersionRead && expansionAwaitsProxyVersion {
				proxyVersion = readProxyVersion()
				// The parents were reduced before the version was known; an expansion must inherit from
				// parents reduced under the version it is published with.
				sources = nil
				for _, group := range wildcardRoutes {
					if model := mapFromModelInfoGroup(groups.rows[group.route], publicCatalog, &groupOptions{proxyVersion: proxyVersion}); model != nil {
						sources = append(sources, wildcardSource{model: *model, deploymentFamilies: group.deploymentFamilies})
					}
				}
			}
			for _, entry := range candidates {
				expanded := mapFromWildcardExpansion(entry, sources, withheldRepairRoutes)
				if expanded == nil {
					continue
				}
				if evidenced := applyWildcardEvidence(*expanded, wildcardRows, publishedWildcardIDs, publicCatalog, proxyVersion); evidenced != nil {
					models = append(models, *evidenced)
				}
			}
		}
	}
	reportWithheldToolRepair(withheldRepairRoutes.list(), options)
	models = deduplicateModels(models)
	if verboseDiscovery() {
		var defaultedWildcards []reducedGroup
		for _, group := range wildcardRoutes {
			if defaultedContextRoutes[group.route] {
				defaultedWildcards = append(defaultedWildcards, group)
			}
		}
		// Exact routes override wildcard templates; a tighter measured parent can also win.
		var defaulted []string
		for _, model := range models {
			var isDefaulted bool
			if _, exact := groups.rows[model.ID]; exact {
				isDefaulted = defaultedContextRoutes[model.ID]
			} else {
				isDefaulted = slices.ContainsFunc(defaultedWildcards, func(group reducedGroup) bool {
					return group.model != nil && group.model.ContextWindow == model.ContextWindow && WildcardMatches(group.route, model.ID)
				})
			}
			if isDefaulted {
				defaulted = append(defaulted, jsonString(model.ID))
			}
		}
		reportDefaultedContext(defaulted, options)
	}
	return &types.DiscoveryResult{Source: types.SourceModelInfo, Models: models, ProxyVersion: proxyVersion}, nil
}

// healthDeployment is one /health endpoint with the /model/info detail correlated to it.
type healthDeployment struct {
	entry      types.ModelInfoEntry
	denyLevels bool
	synthetic  bool
}

// healthDeploymentFor is healthDeployment; nil means the endpoint names no route.
func healthDeploymentFor(detail *types.ModelInfoEntry, fallbackRoute, deploymentID string) *healthDeployment {
	detailRoute := ""
	if detail != nil {
		detailRoute = strings.TrimSpace(detail.ModelName)
	}
	// /health exposes backend litellm_params.model values. Correlated detail owns the public
	// model_name; the health value is only a fallback when detail lacks it.
	route := detailRoute
	if route == "" {
		route = strings.TrimSpace(fallbackRoute)
	}
	if route == "" {
		return nil
	}
	if detail == nil {
		mode := "chat"
		if catalogModel := findCatalogModel(route, ""); catalogModel != nil && catalogModel.API == ai.APIOpenAIResponses {
			mode = "responses"
		}
		details := &types.ModelInfoDetails{Mode: &mode}
		if deploymentID != "" {
			details.ID = deploymentID
		}
		return &healthDeployment{
			entry: types.ModelInfoEntry{ModelName: route, ModelInfo: details},
			// The route name authorizes nothing but transport, which the Pi catalog supplies for this
			// evidence-free entry. Levels stay denied and no catalog metadata is granted.
			denyLevels: true,
			synthetic:  true,
		}
	}
	entry := *detail
	entry.ModelName = route
	details := types.ModelInfoDetails{}
	if detail.ModelInfo != nil {
		details = *detail.ModelInfo
	}
	if strings.TrimSpace(details.ID) == "" && deploymentID != "" {
		details.ID = deploymentID
	}
	entry.ModelInfo = &details
	return &healthDeployment{
		entry: entry,
		// Detail without a public route cannot authorize selectable levels.
		denyLevels: detailRoute == "",
		synthetic:  false,
	}
}

func discoverFromHealth(ctx context.Context, base, apiKey string, options Options) ([]types.DiscoveredModel, error) {
	options.progress("Querying /health endpoint...")
	healthResult, err := FetchJSON[looseRows](ctx, base+"/health", apiKey, options)
	if err != nil {
		return nil, err
	}
	if !healthResult.OK {
		return nil, nil
	}
	type endpoint struct{ model, modelID string }
	var endpoints []endpoint
	for _, row := range healthResult.Data.HealthyEndpoints {
		candidate := endpoint{model: looseString(row, "model"), modelID: looseString(row, "model_id")}
		if strings.TrimSpace(candidate.model) != "" || strings.TrimSpace(candidate.modelID) != "" {
			endpoints = append(endpoints, candidate)
		}
	}
	options.progress(fmt.Sprintf("Discovered %d model endpoints, fetching details...", len(endpoints)))

	var (
		mu        sync.Mutex
		completed int
		next      int
		firstErr  error
	)
	deployments := make([]*healthDeployment, len(endpoints))
	workerCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	worker := func() {
		for {
			mu.Lock()
			if next >= len(endpoints) || firstErr != nil {
				mu.Unlock()
				return
			}
			index := next
			next++
			mu.Unlock()

			current := endpoints[index]
			route := strings.TrimSpace(current.model)
			deploymentID := strings.TrimSpace(current.modelID)
			var detail *types.ModelInfoEntry
			if deploymentID != "" {
				infoResult, err := FetchJSON[types.ModelInfoResponse](
					workerCtx, base+"/model/info?litellm_model_id="+encodeURIComponent(deploymentID), apiKey, options)
				if err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = err
						cancel()
					}
					mu.Unlock()
					return
				}
				if infoResult.OK && len(infoResult.Data.Data) > 0 {
					detail = &infoResult.Data.Data[0]
				}
			}
			mu.Lock()
			completed++
			if completed%10 == 0 || completed == len(endpoints) {
				options.progress(fmt.Sprintf("Fetched %d/%d models...", completed, len(endpoints)))
			}
			deployments[index] = healthDeploymentFor(detail, route, deploymentID)
			mu.Unlock()
		}
	}
	var workers sync.WaitGroup
	for range min(healthDetailConcurrency, len(endpoints)) {
		workers.Add(1)
		go func() {
			defer workers.Done()
			worker()
		}()
	}
	workers.Wait()
	if firstErr != nil {
		if ctx.Err() != nil {
			return nil, context.Cause(ctx)
		}
		return nil, firstErr
	}

	var groups orderedGroups[healthDeployment]
	anyDetailed := false
	for _, deployment := range deployments {
		if deployment == nil || deployment.entry.ModelName == "" {
			continue
		}
		anyDetailed = anyDetailed || !deployment.synthetic
		groups.add(deployment.entry.ModelName, *deployment)
	}
	// TypeScript tests `deployments.some(d => d && !d.synthetic)` over all deployments, including ones
	// without a route; a deployment always has a route, so the two agree.
	var publicCatalog catalog.PublicCatalog
	if anyDetailed {
		publicCatalog, err = loadDiscoveryPublicCatalog(ctx, options)
		if err != nil {
			return nil, err
		}
	}
	entriesOf := func(group []healthDeployment) []types.ModelInfoEntry {
		entries := make([]types.ModelInfoEntry, len(group))
		for i, deployment := range group {
			entries[i] = deployment.entry
		}
		return entries
	}
	var proxyVersion *proxyversion.Version
	for _, route := range groups.routes {
		if publishesVersionGatedTransport(entriesOf(groups.rows[route]), publicCatalog) {
			proxyVersion = ProbeProxyVersion(ctx, base, apiKey, options)
			break
		}
	}
	var incompatibleModeRoutes []string
	ambiguousRoutes, conflictingFamilyRoutes, withheldRepairRoutes := &routeCollector{}, &routeCollector{}, &routeCollector{}
	discovered := []types.DiscoveredModel{}
	for _, route := range groups.routes {
		group := groups.rows[route]
		entries := entriesOf(group)
		if modelgroups.HasMixedIncompatibleDeploymentModes(entries) {
			incompatibleModeRoutes = append(incompatibleModeRoutes, entries[0].ModelName)
		}
		var model *types.DiscoveredModel
		if !slices.ContainsFunc(group, func(deployment healthDeployment) bool { return !deployment.synthetic }) {
			model = mapFromModelsList(types.ModelsListEntry{ID: entries[0].ModelName})
		} else {
			model = mapFromModelInfoGroup(entries, publicCatalog, &groupOptions{
				ambiguousRoutes:         ambiguousRoutes,
				conflictingFamilyRoutes: conflictingFamilyRoutes,
				withheldRepairRoutes:    withheldRepairRoutes,
				denyLevels:              slices.ContainsFunc(group, func(deployment healthDeployment) bool { return deployment.denyLevels }),
				withholdMessages:        true,
				proxyVersion:            proxyVersion,
			})
		}
		if model != nil {
			discovered = append(discovered, *model)
		}
	}
	reportIncompatibleDeploymentModes(incompatibleModeRoutes, options)
	reportAmbiguousCatalogAuthority(ambiguousRoutes.list(), options)
	reportConflictingFamilyEvidence(conflictingFamilyRoutes.list(), options)
	reportWithheldToolRepair(withheldRepairRoutes.list(), options)
	reportWithheldToolRepairForModels(discovered, options)
	return discovered, nil
}

var uriComponentUnescaped = strings.NewReplacer("+", "%20", "%21", "!", "%27", "'", "%28", "(", "%29", ")", "%2A", "*")

// encodeURIComponent is the JavaScript function of the same name.
func encodeURIComponent(value string) string {
	return uriComponentUnescaped.Replace(url.QueryEscape(value))
}
