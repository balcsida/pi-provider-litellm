package discover

import (
	"context"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/catalog"
)

// The initial public-catalog load outlives its budget, so the session shutdown hook must be able to drain it.
func TestLoadDiscoveryPublicCatalogIsDrainedAtShutdown(t *testing.T) {
	release := make(chan struct{})
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		<-release
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("{}"))}, nil
	})}
	budget, modelsDev := 20*time.Millisecond, true
	options := Options{HTTPClient: client}
	options.Timeout, options.ModelsDev = &budget, &modelsDev
	options.ModelsDevCachePath = filepath.Join(t.TempDir(), "models-dev.json")

	if _, err := loadDiscoveryPublicCatalog(context.Background(), options); err != nil {
		t.Fatal(err)
	}
	expired, cancel := context.WithTimeout(context.Background(), budget)
	defer cancel()
	if err := catalog.Drain(expired); err != context.DeadlineExceeded {
		t.Fatalf("Drain with the load in flight = %v, want deadline exceeded", err)
	}
	close(release)
	if err := catalog.Drain(context.Background()); err != nil {
		t.Fatalf("Drain after release = %v", err)
	}
}
