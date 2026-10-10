package bridge

import (
	"context"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/MichaelKinsy/PiG/extensions/sdk"
)

// wrapAuth maps ai.ProviderAuth to sdk.ProviderAuth. Not mappable:
//   - an APIKeyAuth without Resolve, or an OAuthAuth without Login, Refresh or ToAuth, is omitted
//     because the SDK requires those callbacks (sdk/provider.go providerDeclaration);
//   - sdk.AuthContext.Env/FileExists return errors that ai.AuthContext has no slot for; the first
//     one is recorded (hostErrors) and returned from Resolve/Check instead of a false "absent";
//   - ai.AuthInteraction.Notify has no error return, so the first failed notification is recorded
//     and returned from Login once it finishes;
//   - ai.LoginOptions (GetDeviceID) has no SDK counterpart and is passed zero.
func wrapAuth(auth ai.ProviderAuth) sdk.ProviderAuth {
	var out sdk.ProviderAuth
	if a := auth.APIKey; a != nil && a.Resolve != nil {
		out.APIKey = &sdk.APIKeyAuth{
			Name: a.Name,
			Resolve: func(in sdk.APIKeyAuthInput) (*sdk.AuthResult, error) {
				credential, err := decodeCredential(in.Credential)
				if err != nil {
					return nil, err
				}
				var host hostErrors
				result, err := a.Resolve(signal(in.Signal), ai.APIKeyAuthInput{Ctx: host.authContext(in.Ctx), Credential: credential})
				if err = host.or(err); err != nil || result == nil {
					return nil, err
				}
				return sdkAuthResult(result), nil
			},
		}
		if a.Check != nil {
			out.APIKey.Check = func(in sdk.APIKeyAuthInput) (*sdk.AuthCheck, error) {
				credential, err := decodeCredential(in.Credential)
				if err != nil {
					return nil, err
				}
				var host hostErrors
				check, err := a.Check(signal(in.Signal), ai.APIKeyAuthInput{Ctx: host.authContext(in.Ctx), Credential: credential})
				if err = host.or(err); err != nil || check == nil {
					return nil, err
				}
				return &sdk.AuthCheck{Type: string(check.Type), Source: optional(check.Source)}, nil
			}
		}
		if a.Login != nil {
			out.APIKey.Login = func(in sdk.AuthInteraction) (map[string]any, error) {
				var host hostErrors
				credential, err := a.Login(signal(in.Signal), host.interaction(in))
				if err = host.or(err); err != nil {
					return nil, err
				}
				return toMap(credential)
			}
		}
	}
	if o := auth.OAuth; o != nil && o.Login != nil && o.Refresh != nil && o.ToAuth != nil {
		out.OAuth = &sdk.OAuthAuth{
			Name:           o.Name,
			IsSubscription: &o.IsSubscription,
			LoginLabel:     optional(o.LoginLabel),
			Login: func(in sdk.AuthInteraction) (map[string]any, error) {
				var host hostErrors
				credential, err := o.Login(signal(in.Signal), host.interaction(in), ai.LoginOptions{})
				if err = host.or(err); err != nil {
					return nil, err
				}
				return toMap(credential)
			},
			Refresh: func(wire map[string]any, ctx context.Context) (map[string]any, error) {
				credential, err := decodeCredential(wire)
				if err != nil || credential == nil {
					return nil, err
				}
				refreshed, err := o.Refresh(signal(ctx), *credential)
				if err != nil {
					return nil, err
				}
				return toMap(refreshed)
			},
			ToAuth: func(wire map[string]any) (map[string]any, error) {
				credential, err := decodeCredential(wire)
				if err != nil || credential == nil {
					return nil, err
				}
				modelAuth, err := o.ToAuth(*credential)
				if err != nil {
					return nil, err
				}
				return modelAuthMap(modelAuth), nil
			},
		}
	}
	return out
}

func signal(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	return ctx
}

func optional(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// hostErrors records the first failed host callback that the ai callback signatures cannot return.
type hostErrors struct{ first error }

func (h *hostErrors) record(err error) {
	if h.first == nil {
		h.first = err
	}
}

// or prefers err, then the first recorded host error.
func (h *hostErrors) or(err error) error {
	if err != nil {
		return err
	}
	return h.first
}

func (h *hostErrors) authContext(in sdk.AuthContext) ai.AuthContext {
	return ai.AuthContext{
		Env: func(name string) (string, bool) {
			if in.Env == nil {
				return "", false
			}
			value, err := in.Env(name)
			if err != nil {
				h.record(err)
				return "", false
			}
			if value == nil {
				return "", false
			}
			return *value, true
		},
		FileExists: func(path string) bool {
			if in.FileExists == nil {
				return false
			}
			ok, err := in.FileExists(path)
			h.record(err)
			return err == nil && ok
		},
	}
}

func (h *hostErrors) interaction(in sdk.AuthInteraction) ai.AuthInteraction {
	return ai.AuthInteraction{
		Prompt: func(_ context.Context, prompt ai.AuthPrompt) (string, error) {
			wire, err := toMap(prompt)
			if err != nil {
				return "", err
			}
			return in.Prompt(wire)
		},
		Notify: func(event ai.AuthEvent) {
			if in.Notify == nil {
				return
			}
			wire, err := toMap(event)
			if err == nil {
				err = in.Notify(wire)
			}
			if err != nil {
				h.record(err)
			}
		},
	}
}

func sdkAuthResult(r *ai.AuthResult) *sdk.AuthResult {
	out := &sdk.AuthResult{Auth: modelAuthMap(r.Auth), Env: r.Env}
	if r.Source != "" || r.SourcePresent {
		out.Source = &r.Source
	}
	return out
}

// modelAuthMap is the wire form of ai.ModelAuth: apiKey, headers (null deletes), baseUrl.
func modelAuthMap(a ai.ModelAuth) map[string]any {
	out := map[string]any{}
	if a.APIKey != "" {
		out["apiKey"] = a.APIKey
	}
	if a.Headers != nil {
		out["headers"] = a.Headers
	}
	if a.BaseURL != "" {
		out["baseUrl"] = a.BaseURL
	}
	return out
}
