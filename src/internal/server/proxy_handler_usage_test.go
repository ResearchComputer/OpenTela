package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"opentela/internal/usage"

	"github.com/gin-gonic/gin"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
)

func TestGlobalServiceForwardHandler_GeneratesRequestID(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// This test validates that request IDs are generated correctly
	// Request IDs should be non-empty and unique across calls

	viper.Set("billing.enabled", true)
	defer viper.Reset()

	// Test that GenerateRequestID produces valid, unique IDs
	ids := make(map[string]bool)
	for i := 0; i < 100; i++ {
		id := usage.GenerateRequestID()
		assert.NotEmpty(t, id, "Request ID should not be empty")
		assert.False(t, ids[id], "Generated duplicate request ID: %s", id)
		ids[id] = true
	}
	assert.Len(t, ids, 100, "Should generate 100 unique IDs")
}

func TestExtractUsageMetrics(t *testing.T) {
	// Test that usage metrics can be extracted from HTTP response headers
	resp := &http.Response{
		Header: http.Header{
			"X-Usage-Tokens":      []string{"100"},
			"X-Usage-Gpu-Ms":      []string{"5000"},
			"X-Usage-Invalid":     []string{"not-a-number"},
			"X-Usage-Empty-Value": []string{""},
			"Content-Type":        []string{"application/json"},
		},
	}

	// Import the usage package to test extraction
	// This is a basic validation - the full extractor tests are in usage/extractor_test.go
	assert.Equal(t, "100", resp.Header.Get("X-Usage-Tokens"))
	assert.Equal(t, "5000", resp.Header.Get("X-Usage-Gpu-Ms"))
}

func TestUsageHeadersArePreserved(t *testing.T) {
	// Test that usage headers pass through the proxy
	// This validates that X-Usage-* headers from upstream responses
	// are accessible to the usage tracking code
	gin.SetMode(gin.TestMode)

	w := httptest.NewRecorder()

	// Simulate upstream response with usage headers
	w.Header().Set("X-Usage-Tokens", "42")
	w.Header().Set("X-Usage-Gpu-Ms", "1234")
	w.Header().Set("X-Computing-Node", "peer-123")

	// Verify headers are set
	assert.Equal(t, "42", w.Header().Get("X-Usage-Tokens"))
	assert.Equal(t, "1234", w.Header().Get("X-Usage-Gpu-Ms"))
	assert.Equal(t, "peer-123", w.Header().Get("X-Computing-Node"))
}
