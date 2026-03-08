// src/internal/usage/extractor.go
package usage

import (
    "net/http"
    "strconv"
    "strings"
)

// ExtractUsageMetrics parses X-Usage-* headers from an HTTP response
// Header format: X-Usage-Metric-Name: value
// Metric name is lowercased and underscored (e.g., X-Usage-GPU-Ms -> gpu_ms)
func ExtractUsageMetrics(resp *http.Response) (map[string]int64, error) {
    metrics := make(map[string]int64)

    for key, values := range resp.Header {
        if strings.HasPrefix(key, UsageHeaderPrefix) {
            if len(values) == 0 {
                continue
            }

            // Extract metric name (strip "X-Usage-" prefix)
            metricName := strings.TrimPrefix(key, UsageHeaderPrefix)
            // Convert to lowercase and replace hyphens with underscores
            metricName = strings.ToLower(strings.ReplaceAll(metricName, "-", "_"))

            // Parse value
            value, err := strconv.ParseInt(values[0], 10, 64)
            if err != nil {
                // Skip invalid values
                continue
            }

            metrics[metricName] = value
        }
    }

    return metrics, nil
}
