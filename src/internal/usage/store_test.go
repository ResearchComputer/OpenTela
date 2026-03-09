package usage

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestUsageStore_SaveAndGet(t *testing.T) {
	// Create temp directory
	tmpDir := filepath.Join(os.TempDir(), "usage-test-"+time.Now().Format("20060102150405"))
	defer os.RemoveAll(tmpDir)

	store, err := NewUsageStore(tmpDir)
	if err != nil {
		t.Fatalf("NewUsageStore failed: %v", err)
	}
	defer store.Close()

	record := &UsageRecord{
		RequestID:    "test-req-1",
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  1000,
		Timestamp:    time.Now().Unix(),
	}

	// Save
	if err := store.SaveRecord(record); err != nil {
		t.Fatalf("SaveRecord failed: %v", err)
	}

	// Get
	retrieved, err := store.GetRecord("test-req-1")
	if err != nil {
		t.Fatalf("GetRecord failed: %v", err)
	}

	if retrieved.MetricValue != 1000 {
		t.Errorf("Expected MetricValue=1000, got %d", retrieved.MetricValue)
	}
}

func TestUsageStore_GetPendingRecords(t *testing.T) {
	tmpDir := filepath.Join(os.TempDir(), "usage-test-"+time.Now().Format("20060102150405"))
	defer os.RemoveAll(tmpDir)

	store, err := NewUsageStore(tmpDir)
	if err != nil {
		t.Fatalf("NewUsageStore failed: %v", err)
	}
	defer store.Close()

	// Save multiple records
	now := time.Now().Unix()
	for i := 0; i < 3; i++ {
		record := &UsageRecord{
			RequestID:    "test-req-" + string(rune('1'+i)),
			Service:      "llm",
			ConsumerPeer: "head-1",
			ProviderPeer: "worker-1",
			MetricName:   "tokens",
			MetricValue:  int64(1000 * (i + 1)),
			Timestamp:    now,
		}
		if err := store.SaveRecord(record); err != nil {
			t.Fatalf("SaveRecord failed: %v", err)
		}
	}

	// Get pending
	records, err := store.GetPendingRecords("head-1", "worker-1", "llm", "tokens")
	if err != nil {
		t.Fatalf("GetPendingRecords failed: %v", err)
	}

	if len(records) != 3 {
		t.Errorf("Expected 3 records, got %d", len(records))
	}
}
