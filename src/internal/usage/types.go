package usage

// UsageRecord represents a single usage measurement
type UsageRecord struct {
	RequestID    string // Unique identifier for the request
	Service      string // Service name (e.g., "llm", "sandbox")
	ConsumerPeer string // Head node (dispatcher) peer ID
	ProviderPeer string // Worker node (provider) peer ID
	MetricName   string // Metric type (e.g., "tokens", "gpu_ms")
	MetricValue  int64  // Measured value
	Timestamp    int64  // Unix timestamp
	Signature    string // Wallet signature (when billing enabled)
}

// AggregatedUsage represents accumulated usage over a time window
type AggregatedUsage struct {
	PeerID      string // Node that generated this aggregate
	Service     string // Service name
	MetricName  string // Metric type
	TotalValue  int64  // Sum of all metric values
	RecordCount int64  // Number of records aggregated
	WindowStart int64  // Window start time (unix timestamp)
	WindowEnd   int64  // Window end time (unix timestamp)
	Signature   string // Wallet signature
}

// ResolvedUsage represents reconciled usage from dual attestations
type ResolvedUsage struct {
	HeadRecord    *UsageRecord
	WorkerRecord  *UsageRecord
	ResolvedValue int64  // Agreed-upon value (average or original)
	Disputed      bool   // True if difference > threshold
	ResolvedAt    int64  // Unix timestamp
}

// BillingConfig holds runtime billing configuration
type BillingConfig struct {
	Enabled              bool
	ValueThreshold       int64  // In lamports
	MaxIntervalMinutes   int
	DisputeThresholdPct  int    // Percentage
}

const (
	// Header prefix for usage metrics
	UsageHeaderPrefix = "X-Usage-"

	// CRDT namespaces
	CRDTNamespaceUsage       = "/usage"
	CRDTNamespaceDisputed    = "/usage/disputed"
	CRDTNamespacePending     = "/usage/pending"
)
