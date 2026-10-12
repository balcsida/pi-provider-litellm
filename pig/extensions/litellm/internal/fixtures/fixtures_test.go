package fixtures

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPath(t *testing.T) {
	dir := Path(t, "proxy")
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		t.Fatalf("proxy fixtures dir: %v", err)
	}
	if filepath.Base(filepath.Dir(dir)) != "fixtures" {
		t.Fatalf("unexpected path %s", dir)
	}
}
