// src/internal/usage/extractor_test.go
package usage

import (
    "net/http"
    "net/http/httptest"
    "testing"
)

func TestExtractUsageMetrics_Tokens(t *testing.T) {
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("X-Usage-Tokens", "1234")
        w.Header().Set("X-Usage-GPU-Ms", "5000")
        w.WriteHeader(http.StatusOK)
    })
    server := httptest.NewServer(handler)
    defer server.Close()

    resp, err := http.Get(server.URL)
    if err != nil {
        t.Fatal(err)
    }
    defer resp.Body.Close()

    metrics, err := ExtractUsageMetrics(resp)
    if err != nil {
        t.Fatalf("ExtractUsageMetrics failed: %v", err)
    }

    if metrics["tokens"] != 1234 {
        t.Errorf("Expected tokens=1234, got %d", metrics["tokens"])
    }
    if metrics["gpu_ms"] != 5000 {
        t.Errorf("Expected gpu_ms=5000, got %d", metrics["gpu_ms"])
    }
}

func TestExtractUsageMetrics_NoHeaders(t *testing.T) {
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
    })
    server := httptest.NewServer(handler)
    defer server.Close()

    resp, err := http.Get(server.URL)
    if err != nil {
        t.Fatal(err)
    }
    defer resp.Body.Close()

    metrics, err := ExtractUsageMetrics(resp)
    if err != nil {
        t.Fatalf("ExtractUsageMetrics failed: %v", err)
    }

    if len(metrics) != 0 {
        t.Errorf("Expected no metrics, got %v", metrics)
    }
}
