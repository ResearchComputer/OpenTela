package solana

import (
	"context"
	"testing"

	"opentela/internal/usage"
)

func TestSubmitSettlement_BillingDisabled(t *testing.T) {
	ctx := context.Background()
	records := []*usage.ResolvedUsage{
		{
			ResolvedValue: 1000,
			Disputed:      false,
		},
	}

	sig, err := SubmitSettlement(ctx, records)
	if err != nil {
		t.Fatalf("SubmitSettlement failed: %v", err)
	}

	if sig != "" {
		t.Errorf("Expected empty signature when billing disabled, got %s", sig)
	}
}
