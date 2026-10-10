// Package fixtures locates the repository's shared tests/fixtures directory.
package fixtures

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Path resolves rel under <repo>/tests/fixtures and fails the test when it does not exist.
func Path(t testing.TB, rel string) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("fixtures: cannot locate source file")
	}
	// fixtures → internal → litellm → extensions → pig → repository root
	root := filepath.Join(filepath.Dir(file), "..", "..", "..", "..", "..")
	p := filepath.Clean(filepath.Join(root, "tests", "fixtures", filepath.FromSlash(rel)))
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("fixtures: %v", err)
	}
	return p
}
