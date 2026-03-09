package usage

import (
    "testing"
    "time"
)

func TestAggregator_ShouldFlush_ValueThreshold(t *testing.T) {
    agg := NewAggregator(&BillingConfig{
        ValueThreshold:     10000000, // ~$10 worth
        MaxIntervalMinutes: 60,
    })

    // Add records totaling less than threshold
    for i := 0; i < 10; i++ {
        agg.AddRecord("peer-1", "llm", "tokens", 1000)
    }

    if agg.ShouldFlush("peer-1", "llm", "tokens") {
        t.Error("Should not flush - value threshold not reached")
    }

    // Add more to exceed threshold
    agg.AddRecord("peer-1", "llm", "tokens", 10000000)

    if !agg.ShouldFlush("peer-1", "llm", "tokens") {
        t.Error("Should flush - value threshold exceeded")
    }
}

func TestAggregator_ShouldFlush_TimeCeiling(t *testing.T) {
    agg := NewAggregator(&BillingConfig{
        ValueThreshold:     10000000,
        MaxIntervalMinutes: 1, // 1 minute for testing
    })

    agg.AddRecord("peer-1", "llm", "tokens", 100)
    agg.SetWindowStart("peer-1", "llm", "tokens", time.Now().Add(-2*time.Minute).Unix())

    if !agg.ShouldFlush("peer-1", "llm", "tokens") {
        t.Error("Should flush - time ceiling exceeded")
    }
}

func TestAggregator_BuildAggregate(t *testing.T) {
    agg := NewAggregator(&BillingConfig{
        ValueThreshold:     10000000,
        MaxIntervalMinutes: 60,
    })

    agg.AddRecord("peer-1", "llm", "tokens", 1000)
    agg.AddRecord("peer-1", "llm", "tokens", 2000)
    agg.AddRecord("peer-1", "llm", "tokens", 3000)

    agg.BuildAggregate("peer-1", "llm", "tokens")

    // Should be reset after building
    if agg.GetValue("peer-1", "llm", "tokens") != 0 {
        t.Error("Aggregate should be reset after BuildAggregate")
    }
}
