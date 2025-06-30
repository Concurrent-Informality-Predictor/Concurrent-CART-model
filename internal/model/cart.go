package model

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"

	"go-ml-informalidad/internal/redis"
)

// NormalizationParams stores normalization parameters for consistent scaling
type NormalizationParams struct {
	Mins []float64 `json:"mins"`
	Maxs []float64 `json:"maxs"`
}

// TrainingData represents the dataset for training
type TrainingData struct {
	Features [][]float64            `json:"features"`
	Labels   []float64              `json:"labels"`
	Metadata map[string]interface{} `json:"metadata"`
}

// PredictionRequest represents a request for prediction
type PredictionRequest struct {
	ID        string    `json:"id"`
	Features  []float64 `json:"features"`
	Timestamp time.Time `json:"timestamp"`
}

// PredictionResult represents the result of a prediction
type PredictionResult struct {
	ID          string    `json:"id"`
	Probability float64   `json:"probability"`
	Class       int       `json:"class"`
	Confidence  float64   `json:"confidence"`
	Timestamp   time.Time `json:"timestamp"`
}

// Node represents a node in the CART decision tree
type Node struct {
	FeatureIndex int             `json:"feature_index"`
	Threshold    float64         `json:"threshold"`
	Left         *Node           `json:"left"`
	Right        *Node           `json:"right"`
	Value        float64         `json:"value"`
	IsLeaf       bool            `json:"is_leaf"`
	Samples      int             `json:"samples"`
	Gini         float64         `json:"gini"`
	ClassCounts  map[float64]int `json:"class_counts"` // Store class distribution in leaf nodes
}

// CART implements a concurrent CART decision tree model
type CART struct {
	root              *Node
	maxDepth          int
	minSamples        int
	minImpurity       float64
	featureNames      []string
	normalizationMins []float64 // Store normalization parameters
	normalizationMaxs []float64 // Store normalization parameters
	mutex             sync.RWMutex
	trained           bool
}

// NewCART creates a new CART decision tree model with concurrent processing
func NewCART() *CART {
	return &CART{
		maxDepth:    15, // Reduced for faster training
		minSamples:  50, // Increased for better generalization
		minImpurity: 1e-6,
		trained:     false,
	}
}

// Split represents a potential split in the tree
type Split struct {
	FeatureIndex int
	Threshold    float64
	GiniLeft     float64
	GiniRight    float64
	GiniGain     float64
	LeftIndices  []int
	RightIndices []int
}

// Train trains the CART model using the provided features and labels
func (c *CART) Train(features [][]float64, labels []float64) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if len(features) == 0 || len(labels) == 0 {
		return fmt.Errorf("empty training data")
	}

	if len(features) != len(labels) {
		return fmt.Errorf("features and labels length mismatch")
	}

	startTime := time.Now()
	log.Printf("🚀 Starting CART training with %d samples and %d features at %s",
		len(features), len(features[0]), startTime.Format("15:04:05"))

	// Create indices for all samples
	indices := make([]int, len(features))
	for i := range indices {
		indices[i] = i
	}

	// Build the tree with progress tracking
	log.Printf("📊 Building decision tree (max_depth=%d, min_samples=%d)...", c.maxDepth, c.minSamples)
	c.root = c.buildTreeWithProgress(features, labels, indices, 0, startTime)
	c.trained = true

	elapsed := time.Since(startTime)
	log.Printf("✅ CART training completed successfully in %v", elapsed)
	log.Printf("📈 Tree statistics: depth=%d, leaves=%d",
		c.getTreeDepth(c.root), c.getLeafCount(c.root))

	return nil
}

// TrainWithData trains the CART model using training data that includes normalization parameters
func (c *CART) TrainWithData(trainingData *TrainingData) error {
	// Extract normalization parameters from metadata
	if params, exists := trainingData.Metadata["normalization_params"]; exists {
		if normParams, ok := params.(*NormalizationParams); ok {
			c.normalizationMins = normParams.Mins
			c.normalizationMaxs = normParams.Maxs
			log.Printf("🔢 Stored normalization parameters for prediction consistency")
		}
	}

	// Call the original training method
	return c.Train(trainingData.Features, trainingData.Labels)
}

// buildTreeWithProgress recursively builds the decision tree with progress tracking
func (c *CART) buildTreeWithProgress(features [][]float64, labels []float64, indices []int, depth int, startTime time.Time) *Node {
	// Log progress every few levels or when processing large datasets
	if depth <= 5 && len(indices) > 1000 {
		c.LogTrainingProgress(startTime, depth, len(indices))
	}

	return c.buildTree(features, labels, indices, depth)
}

// buildTree recursively builds the decision tree
func (c *CART) buildTree(features [][]float64, labels []float64, indices []int, depth int) *Node {
	// Calculate Gini impurity for current node
	gini := c.calculateGini(labels, indices)
	samples := len(indices)

	// Create leaf node if stopping criteria are met
	if depth >= c.maxDepth || samples < c.minSamples || gini < c.minImpurity {
		value := c.calculateMajorityClass(labels, indices)
		classCounts := c.calculateClassCounts(labels, indices)
		return &Node{
			IsLeaf:      true,
			Value:       value,
			Samples:     samples,
			Gini:        gini,
			ClassCounts: classCounts,
		}
	}

	// Find best split
	bestSplit := c.findBestSplit(features, labels, indices)
	if bestSplit == nil || bestSplit.GiniGain <= 0 {
		// No good split found, create leaf
		value := c.calculateMajorityClass(labels, indices)
		classCounts := c.calculateClassCounts(labels, indices)
		return &Node{
			IsLeaf:      true,
			Value:       value,
			Samples:     samples,
			Gini:        gini,
			ClassCounts: classCounts,
		}
	}

	// Create internal node
	node := &Node{
		FeatureIndex: bestSplit.FeatureIndex,
		Threshold:    bestSplit.Threshold,
		IsLeaf:       false,
		Samples:      samples,
		Gini:         gini,
	}

	// Recursively build left and right subtrees
	node.Left = c.buildTree(features, labels, bestSplit.LeftIndices, depth+1)
	node.Right = c.buildTree(features, labels, bestSplit.RightIndices, depth+1)

	return node
}

// findBestSplit finds the best split for the given data using concurrent processing
func (c *CART) findBestSplit(features [][]float64, labels []float64, indices []int) *Split {
	if len(indices) < 2 {
		return nil
	}

	numFeatures := len(features[0])
	var bestSplit *Split
	bestGain := 0.0

	// Log progress for large datasets
	if len(indices) > 10000 {
		log.Printf("🔍 Finding best split for %d samples across %d features (concurrent)...", len(indices), numFeatures)
	}

	// Channel for collecting splits from concurrent goroutines
	splitChan := make(chan *Split, numFeatures)

	// Process features concurrently
	var wg sync.WaitGroup
	for featureIndex := 0; featureIndex < numFeatures; featureIndex++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			bestFeatureSplit := c.findBestSplitForFeature(features, labels, indices, idx)
			if bestFeatureSplit != nil {
				splitChan <- bestFeatureSplit
			}
		}(featureIndex)
	}

	// Close channel when all goroutines are done
	go func() {
		wg.Wait()
		close(splitChan)
	}()

	// Collect results and find the best split
	for split := range splitChan {
		if split != nil && split.GiniGain > bestGain {
			bestGain = split.GiniGain
			bestSplit = split
		}
	}

	return bestSplit
}

// findBestSplitForFeature finds the best split for a specific feature
func (c *CART) findBestSplitForFeature(features [][]float64, labels []float64, indices []int, featureIndex int) *Split {
	// Get unique values for this feature
	values := make([]float64, len(indices))
	for i, idx := range indices {
		values[i] = features[idx][featureIndex]
	}

	// Sort values to find potential thresholds
	sort.Float64s(values)

	var bestSplit *Split
	bestGain := 0.0

	// Try splits between consecutive unique values
	for i := 0; i < len(values)-1; i++ {
		if values[i] == values[i+1] {
			continue // Skip identical values
		}

		threshold := (values[i] + values[i+1]) / 2.0
		split := c.evaluateSplit(features, labels, indices, featureIndex, threshold)

		if split != nil && split.GiniGain > bestGain {
			bestGain = split.GiniGain
			bestSplit = split
		}
	}

	return bestSplit
}

// evaluateSplit evaluates a potential split
func (c *CART) evaluateSplit(features [][]float64, labels []float64, indices []int, featureIndex int, threshold float64) *Split {
	var leftIndices, rightIndices []int

	// Split the data
	for _, idx := range indices {
		if features[idx][featureIndex] <= threshold {
			leftIndices = append(leftIndices, idx)
		} else {
			rightIndices = append(rightIndices, idx)
		}
	}

	// Check if split is valid
	if len(leftIndices) == 0 || len(rightIndices) == 0 {
		return nil
	}

	// Calculate Gini impurities
	giniLeft := c.calculateGini(labels, leftIndices)
	giniRight := c.calculateGini(labels, rightIndices)

	// Calculate weighted Gini impurity after split
	totalSamples := float64(len(indices))
	leftWeight := float64(len(leftIndices)) / totalSamples
	rightWeight := float64(len(rightIndices)) / totalSamples
	weightedGini := leftWeight*giniLeft + rightWeight*giniRight

	// Calculate Gini gain (reduction in impurity)
	parentGini := c.calculateGini(labels, indices)
	giniGain := parentGini - weightedGini

	return &Split{
		FeatureIndex: featureIndex,
		Threshold:    threshold,
		GiniLeft:     giniLeft,
		GiniRight:    giniRight,
		GiniGain:     giniGain,
		LeftIndices:  leftIndices,
		RightIndices: rightIndices,
	}
}

// calculateGini calculates the Gini impurity for given indices
func (c *CART) calculateGini(labels []float64, indices []int) float64 {
	if len(indices) == 0 {
		return 0.0
	}

	// Count class frequencies
	classCount := make(map[float64]int)
	for _, idx := range indices {
		classCount[labels[idx]]++
	}

	// Calculate Gini impurity: 1 - sum(p_i^2)
	gini := 1.0
	total := float64(len(indices))
	for _, count := range classCount {
		p := float64(count) / total
		gini -= p * p
	}

	return gini
}

// calculateMajorityClass returns the majority class for given indices
func (c *CART) calculateMajorityClass(labels []float64, indices []int) float64 {
	if len(indices) == 0 {
		return 0.0
	}

	classCount := make(map[float64]int)
	for _, idx := range indices {
		classCount[labels[idx]]++
	}

	var majorityClass float64
	maxCount := 0
	for class, count := range classCount {
		if count > maxCount {
			maxCount = count
			majorityClass = class
		}
	}

	return majorityClass
}

// Predict makes a prediction for given features
func (c *CART) Predict(features []float64) (*PredictionResult, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if !c.trained {
		return nil, fmt.Errorf("model not trained yet")
	}

	if c.root == nil {
		return nil, fmt.Errorf("model root is nil")
	}

	// Find the leaf node for this prediction and get class probabilities
	leafNode := c.findLeafNode(c.root, features)

	// Calculate class probabilities based on the leaf node's samples
	probability, class := c.calculateClassProbabilities(leafNode)

	// Calculate confidence based on node purity
	confidence := 1.0 - leafNode.Gini

	return &PredictionResult{
		Probability: probability,
		Class:       class,
		Confidence:  confidence,
		Timestamp:   time.Now(),
	}, nil
}

// StartTrainingWorker starts a goroutine that handles training requests
func (c *CART) StartTrainingWorker(ctx context.Context, trainingChan <-chan *TrainingData, redisClient *redis.Client) {
	log.Println("CART training worker started")

	for {
		select {
		case <-ctx.Done():
			log.Println("CART training worker stopped")
			return
		case data := <-trainingChan:
			log.Printf("Received training data with %d samples", len(data.Features))

			// Train the model
			if err := c.Train(data.Features, data.Labels); err != nil {
				log.Printf("CART training error: %v", err)
				continue
			}

			// Save model to Redis
			if err := c.saveModelToRedis(redisClient); err != nil {
				log.Printf("Error saving CART model to Redis: %v", err)
			} else {
				log.Println("CART model saved to Redis successfully")
			}
		}
	}
}

// StartPredictionWorker starts a goroutine that handles prediction requests
func (c *CART) StartPredictionWorker(ctx context.Context, workerID int,
	predictionChan <-chan *PredictionRequest, resultChan chan<- *PredictionResult,
	redisClient *redis.Client) {

	log.Printf("CART prediction worker %d started", workerID)

	for {
		select {
		case <-ctx.Done():
			log.Printf("CART prediction worker %d stopped", workerID)
			return
		case req := <-predictionChan:
			// Load model from Redis if not trained locally
			if !c.trained {
				if err := c.loadModelFromRedis(redisClient); err != nil {
					log.Printf("Worker %d: Error loading CART model from Redis: %v", workerID, err)
					continue
				}
			}

			// Make prediction
			result, err := c.Predict(req.Features)
			if err != nil {
				log.Printf("Worker %d: CART prediction error: %v", workerID, err)
				continue
			}

			result.ID = req.ID
			result.Timestamp = time.Now()

			log.Printf("Worker %d: CART prediction for %s - Class: %d, Confidence: %.4f",
				workerID, req.ID, result.Class, result.Confidence)

			// Send result
			select {
			case resultChan <- result:
			case <-ctx.Done():
				return
			}
		}
	}
}

// saveModelToRedis saves the trained model to Redis
func (c *CART) saveModelToRedis(redisClient *redis.Client) error {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if !c.trained {
		return fmt.Errorf("CART model not trained")
	}

	// Convert the tree to a JSON-serializable format
	serializedRoot := c.serializeNode(c.root)

	modelData := map[string]interface{}{
		"root":               serializedRoot,
		"max_depth":          c.maxDepth,
		"min_samples":        c.minSamples,
		"min_impurity":       c.minImpurity,
		"trained":            c.trained,
		"normalization_mins": c.normalizationMins,
		"normalization_maxs": c.normalizationMaxs,
		"timestamp":          time.Now().Unix(),
	}

	data, err := json.Marshal(modelData)
	if err != nil {
		return fmt.Errorf("error marshaling CART model data: %v", err)
	}

	return redisClient.Set("cart_model", string(data))
}

// loadModelFromRedis loads the trained model from Redis
func (c *CART) loadModelFromRedis(redisClient *redis.Client) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	data, err := redisClient.Get("cart_model")
	if err != nil {
		return fmt.Errorf("error getting CART model from Redis: %v", err)
	}

	var modelData map[string]interface{}
	if err := json.Unmarshal([]byte(data), &modelData); err != nil {
		return fmt.Errorf("error unmarshaling CART model data: %v", err)
	}

	// Reconstruct the tree from JSON using the serializable format
	rootData, _ := json.Marshal(modelData["root"])
	var serializedRoot SerializableNode
	if err := json.Unmarshal(rootData, &serializedRoot); err != nil {
		return fmt.Errorf("error reconstructing tree: %v", err)
	}

	// Convert back to the internal Node format
	c.root = c.deserializeNode(&serializedRoot)
	c.maxDepth = int(modelData["max_depth"].(float64))
	c.minSamples = int(modelData["min_samples"].(float64))
	c.minImpurity = modelData["min_impurity"].(float64)
	c.trained = modelData["trained"].(bool)

	// Load normalization parameters if available
	if normMins, exists := modelData["normalization_mins"]; exists && normMins != nil {
		if normMinsSlice, ok := normMins.([]interface{}); ok {
			c.normalizationMins = make([]float64, len(normMinsSlice))
			for i, v := range normMinsSlice {
				c.normalizationMins[i] = v.(float64)
			}
		}
	}

	if normMaxs, exists := modelData["normalization_maxs"]; exists && normMaxs != nil {
		if normMaxsSlice, ok := normMaxs.([]interface{}); ok {
			c.normalizationMaxs = make([]float64, len(normMaxsSlice))
			for i, v := range normMaxsSlice {
				c.normalizationMaxs[i] = v.(float64)
			}
		}
	}

	if len(c.normalizationMins) > 0 && len(c.normalizationMaxs) > 0 {
		log.Printf("🔢 Loaded normalization parameters from Redis")
	} else {
		log.Printf("⚠️ No normalization parameters found in Redis")
	}

	log.Println("CART model loaded from Redis successfully")
	return nil
}

// LoadModelFromRedis loads the trained model from Redis (public method)
func (c *CART) LoadModelFromRedis(redisClient *redis.Client) error {
	return c.loadModelFromRedis(redisClient)
}

// SaveModelToRedis saves the trained model to Redis (public method)
func (c *CART) SaveModelToRedis(redisClient *redis.Client) error {
	return c.saveModelToRedis(redisClient)
}

// GetModelInfo returns information about the current model
func (c *CART) GetModelInfo() map[string]interface{} {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	info := map[string]interface{}{
		"model_type":   "CART Decision Tree",
		"trained":      c.trained,
		"max_depth":    c.maxDepth,
		"min_samples":  c.minSamples,
		"min_impurity": c.minImpurity,
	}

	if c.trained && c.root != nil {
		info["tree_depth"] = c.getTreeDepth(c.root)
		info["leaf_count"] = c.getLeafCount(c.root)
	}

	return info
}

// getTreeDepth calculates the depth of the tree
func (c *CART) getTreeDepth(node *Node) int {
	if node == nil || node.IsLeaf {
		return 0
	}

	leftDepth := c.getTreeDepth(node.Left)
	rightDepth := c.getTreeDepth(node.Right)

	return 1 + int(math.Max(float64(leftDepth), float64(rightDepth)))
}

// getLeafCount counts the number of leaf nodes
func (c *CART) getLeafCount(node *Node) int {
	if node == nil {
		return 0
	}
	if node.IsLeaf {
		return 1
	}
	return c.getLeafCount(node.Left) + c.getLeafCount(node.Right)
}

// LogTrainingProgress logs the current training progress
func (c *CART) LogTrainingProgress(startTime time.Time, currentDepth int, totalSamples int) {
	elapsed := time.Since(startTime)

	// Estimate progress based on depth (rough estimation)
	progressPercent := float64(currentDepth) / float64(c.maxDepth) * 100
	if progressPercent > 100 {
		progressPercent = 100
	}

	// Estimate remaining time (very rough)
	if progressPercent > 0 {
		estimatedTotal := time.Duration(float64(elapsed) / progressPercent * 100)
		remaining := estimatedTotal - elapsed

		log.Printf("📊 Progress: %.1f%% | Depth: %d/%d | Elapsed: %v | ETA: %v",
			progressPercent, currentDepth, c.maxDepth, elapsed, remaining)
	} else {
		log.Printf("📊 Progress: %.1f%% | Depth: %d/%d | Elapsed: %v",
			progressPercent, currentDepth, c.maxDepth, elapsed)
	}
}

// TrainWithSample trains the CART model using a sample of the data for faster training
func (c *CART) TrainWithSample(features [][]float64, labels []float64, maxSamples int) error {
	if maxSamples > 0 && len(features) > maxSamples {
		log.Printf("🎯 Using sample of %d from %d total samples for faster training", maxSamples, len(features))

		// Create random sample indices
		sampleIndices := make([]int, maxSamples)
		for i := 0; i < maxSamples; i++ {
			sampleIndices[i] = i
		}

		// Create sampled data
		sampledFeatures := make([][]float64, maxSamples)
		sampledLabels := make([]float64, maxSamples)

		for i, idx := range sampleIndices {
			sampledFeatures[i] = features[idx]
			sampledLabels[i] = labels[idx]
		}

		return c.Train(sampledFeatures, sampledLabels)
	}

	return c.Train(features, labels)
}

// findLeafNode finds the leaf node that corresponds to the given features
func (c *CART) findLeafNode(node *Node, features []float64) *Node {
	if node.IsLeaf {
		return node
	}

	if features[node.FeatureIndex] <= node.Threshold {
		return c.findLeafNode(node.Left, features)
	} else {
		return c.findLeafNode(node.Right, features)
	}
}

// calculateClassProbabilities calculates class probabilities and returns the predicted class
func (c *CART) calculateClassProbabilities(leafNode *Node) (float64, int) {
	// Use actual class distribution from the leaf node
	if len(leafNode.ClassCounts) == 0 {
		// Fallback to simple majority class
		return leafNode.Value, int(leafNode.Value)
	}

	totalSamples := 0
	for _, count := range leafNode.ClassCounts {
		totalSamples += count
	}

	if totalSamples == 0 {
		return 0.5, 0 // Default fallback
	}

	// Calculate probability of class 1 (formal employment)
	class1Count := leafNode.ClassCounts[1.0]
	probability := float64(class1Count) / float64(totalSamples)

	// Determine predicted class
	var predictedClass int
	if probability >= 0.5 {
		predictedClass = 1
	} else {
		predictedClass = 0
	}

	log.Printf("🔍 Leaf node prediction: Class counts: %v, Total: %d, P(class=1): %.3f, Predicted: %d",
		leafNode.ClassCounts, totalSamples, probability, predictedClass)

	return probability, predictedClass
}

// calculateClassCounts calculates the distribution of classes for given indices
func (c *CART) calculateClassCounts(labels []float64, indices []int) map[float64]int {
	classCounts := make(map[float64]int)
	for _, idx := range indices {
		classCounts[labels[idx]]++
	}
	return classCounts
}

// ValidationResult contains comprehensive model evaluation metrics
type ValidationResult struct {
	TrainAccuracy     float64            `json:"train_accuracy"`
	TestAccuracy      float64            `json:"test_accuracy"`
	TrainPrecision    float64            `json:"train_precision"`
	TestPrecision     float64            `json:"test_precision"`
	TrainRecall       float64            `json:"train_recall"`
	TestRecall        float64            `json:"test_recall"`
	TrainF1Score      float64            `json:"train_f1_score"`
	TestF1Score       float64            `json:"test_f1_score"`
	AccuracyDiff      float64            `json:"accuracy_diff"`
	OverfittingRisk   string             `json:"overfitting_risk"`
	TreeDepth         int                `json:"tree_depth"`
	LeafCount         int                `json:"leaf_count"`
	TrainSamples      int                `json:"train_samples"`
	TestSamples       int                `json:"test_samples"`
	ConfusionMatrix   map[string]int     `json:"confusion_matrix"`
	ClassDistribution map[string]float64 `json:"class_distribution"`
}

// TrainWithValidation trains the model with train/test split and comprehensive evaluation
func (c *CART) TrainWithValidation(features [][]float64, labels []float64, testRatio float64) (*ValidationResult, error) {
	if testRatio <= 0 || testRatio >= 1 {
		return nil, fmt.Errorf("test ratio must be between 0 and 1, got %.2f", testRatio)
	}

	log.Printf("🔄 Starting training with validation (test ratio: %.1f%%)", testRatio*100)

	// Shuffle and split data randomly
	trainFeatures, trainLabels, testFeatures, testLabels := c.splitDataRandomly(features, labels, testRatio)

	log.Printf("📊 Data split: Training=%d samples, Testing=%d samples",
		len(trainFeatures), len(testFeatures))

	// Train the model on training data
	startTime := time.Now()
	err := c.Train(trainFeatures, trainLabels)
	if err != nil {
		return nil, fmt.Errorf("training failed: %v", err)
	}

	trainingTime := time.Since(startTime)
	log.Printf("⏱️ Training completed in %v", trainingTime)

	// Evaluate on both training and test sets
	log.Printf("📈 Evaluating model performance...")

	trainMetrics := c.calculateDetailedMetrics(trainFeatures, trainLabels)
	testMetrics := c.calculateDetailedMetrics(testFeatures, testLabels)

	// Calculate overfitting indicators
	accuracyDiff := trainMetrics.Accuracy - testMetrics.Accuracy

	var overfittingRisk string
	if accuracyDiff > 0.15 { // 15% difference
		overfittingRisk = "High"
	} else if accuracyDiff > 0.08 { // 8% difference
		overfittingRisk = "Medium"
	} else if accuracyDiff < -0.05 { // Model performs worse on training (underfitting)
		overfittingRisk = "Underfitting"
	} else {
		overfittingRisk = "Low"
	}

	// Calculate class distribution
	classDistribution := c.calculateClassDistribution(labels)

	result := &ValidationResult{
		TrainAccuracy:     trainMetrics.Accuracy,
		TestAccuracy:      testMetrics.Accuracy,
		TrainPrecision:    trainMetrics.Precision,
		TestPrecision:     testMetrics.Precision,
		TrainRecall:       trainMetrics.Recall,
		TestRecall:        testMetrics.Recall,
		TrainF1Score:      trainMetrics.F1Score,
		TestF1Score:       testMetrics.F1Score,
		AccuracyDiff:      accuracyDiff,
		OverfittingRisk:   overfittingRisk,
		TreeDepth:         c.getTreeDepth(c.root),
		LeafCount:         c.getLeafCount(c.root),
		TrainSamples:      len(trainFeatures),
		TestSamples:       len(testFeatures),
		ConfusionMatrix:   testMetrics.ConfusionMatrix,
		ClassDistribution: classDistribution,
	}

	// Log comprehensive results
	c.logValidationResults(result)

	return result, nil
}

// splitDataRandomly splits data into train/test sets randomly
func (c *CART) splitDataRandomly(features [][]float64, labels []float64, testRatio float64) (
	[][]float64, []float64, [][]float64, []float64) {

	totalSamples := len(features)
	testSize := int(float64(totalSamples) * testRatio)

	// Create random indices
	indices := make([]int, totalSamples)
	for i := range indices {
		indices[i] = i
	}

	// Simple shuffle using current time as seed
	seed := time.Now().UnixNano()
	for i := len(indices) - 1; i > 0; i-- {
		seed = seed*1103515245 + 12345 // Linear congruential generator
		j := int(seed) % (i + 1)
		if j < 0 {
			j = -j
		}
		indices[i], indices[j] = indices[j], indices[i]
	}

	// Split indices
	testIndices := indices[:testSize]
	trainIndices := indices[testSize:]

	// Create train sets
	trainFeatures := make([][]float64, len(trainIndices))
	trainLabels := make([]float64, len(trainIndices))
	for i, idx := range trainIndices {
		trainFeatures[i] = features[idx]
		trainLabels[i] = labels[idx]
	}

	// Create test sets
	testFeatures := make([][]float64, len(testIndices))
	testLabels := make([]float64, len(testIndices))
	for i, idx := range testIndices {
		testFeatures[i] = features[idx]
		testLabels[i] = labels[idx]
	}

	return trainFeatures, trainLabels, testFeatures, testLabels
}

// DetailedMetrics provides comprehensive model evaluation
type DetailedMetrics struct {
	Accuracy        float64        `json:"accuracy"`
	Precision       float64        `json:"precision"`
	Recall          float64        `json:"recall"`
	F1Score         float64        `json:"f1_score"`
	ConfusionMatrix map[string]int `json:"confusion_matrix"`
}

// calculateDetailedMetrics calculates comprehensive metrics for model evaluation
func (c *CART) calculateDetailedMetrics(features [][]float64, labels []float64) *DetailedMetrics {
	tp, fp, tn, fn := 0, 0, 0, 0

	for i := 0; i < len(features); i++ {
		prediction, err := c.Predict(features[i])
		if err != nil {
			continue
		}

		predicted := prediction.Class
		actual := int(labels[i])

		if predicted == 1 && actual == 1 {
			tp++
		} else if predicted == 1 && actual == 0 {
			fp++
		} else if predicted == 0 && actual == 0 {
			tn++
		} else if predicted == 0 && actual == 1 {
			fn++
		}
	}

	// Calculate metrics
	total := tp + fp + tn + fn
	accuracy := float64(tp+tn) / float64(total)

	var precision, recall, f1Score float64

	if tp+fp > 0 {
		precision = float64(tp) / float64(tp+fp)
	}

	if tp+fn > 0 {
		recall = float64(tp) / float64(tp+fn)
	}

	if precision+recall > 0 {
		f1Score = 2 * (precision * recall) / (precision + recall)
	}

	return &DetailedMetrics{
		Accuracy:  accuracy,
		Precision: precision,
		Recall:    recall,
		F1Score:   f1Score,
		ConfusionMatrix: map[string]int{
			"true_positive":  tp,
			"false_positive": fp,
			"true_negative":  tn,
			"false_negative": fn,
		},
	}
}

// calculateClassDistribution calculates the distribution of classes in the dataset
func (c *CART) calculateClassDistribution(labels []float64) map[string]float64 {
	classCounts := make(map[float64]int)
	total := len(labels)

	for _, label := range labels {
		classCounts[label]++
	}

	distribution := make(map[string]float64)
	for class, count := range classCounts {
		if class == 0.0 {
			distribution["informal"] = float64(count) / float64(total)
		} else if class == 1.0 {
			distribution["formal"] = float64(count) / float64(total)
		}
	}

	return distribution
}

// logValidationResults logs comprehensive validation results
func (c *CART) logValidationResults(result *ValidationResult) {
	log.Printf("🎯 =================== MODEL EVALUATION RESULTS ===================")
	log.Printf("📊 Dataset Information:")
	log.Printf("   Training Samples: %d", result.TrainSamples)
	log.Printf("   Testing Samples: %d", result.TestSamples)
	log.Printf("   Class Distribution: Formal=%.1f%%, Informal=%.1f%%",
		result.ClassDistribution["formal"]*100, result.ClassDistribution["informal"]*100)

	log.Printf("🌳 Tree Structure:")
	log.Printf("   Depth: %d", result.TreeDepth)
	log.Printf("   Leaves: %d", result.LeafCount)

	log.Printf("📈 Performance Metrics:")
	log.Printf("   Training Accuracy: %.4f (%.2f%%)", result.TrainAccuracy, result.TrainAccuracy*100)
	log.Printf("   Testing Accuracy:  %.4f (%.2f%%)", result.TestAccuracy, result.TestAccuracy*100)
	log.Printf("   Accuracy Difference: %.4f", result.AccuracyDiff)

	log.Printf("🎯 Detailed Metrics (Test Set):")
	log.Printf("   Precision: %.4f", result.TestPrecision)
	log.Printf("   Recall:    %.4f", result.TestRecall)
	log.Printf("   F1-Score:  %.4f", result.TestF1Score)

	log.Printf("📋 Confusion Matrix (Test Set):")
	cm := result.ConfusionMatrix
	log.Printf("   True Positive:  %d", cm["true_positive"])
	log.Printf("   False Positive: %d", cm["false_positive"])
	log.Printf("   True Negative:  %d", cm["true_negative"])
	log.Printf("   False Negative: %d", cm["false_negative"])

	log.Printf("⚠️ Overfitting Analysis:")
	log.Printf("   Risk Level: %s", result.OverfittingRisk)

	switch result.OverfittingRisk {
	case "High":
		log.Printf("   🔴 HIGH OVERFITTING DETECTED!")
		log.Printf("   Recommendations:")
		log.Printf("     - Reduce tree depth (current: %d → try: %d)", result.TreeDepth, result.TreeDepth-3)
		log.Printf("     - Increase min_samples (current: 50 → try: 100)")
		log.Printf("     - Consider pruning")
	case "Medium":
		log.Printf("   🟡 MODERATE OVERFITTING")
		log.Printf("   Recommendations:")
		log.Printf("     - Slightly reduce complexity")
		log.Printf("     - Monitor with more data")
	case "Underfitting":
		log.Printf("   🔵 UNDERFITTING DETECTED!")
		log.Printf("   Recommendations:")
		log.Printf("     - Increase tree depth (current: %d → try: %d)", result.TreeDepth, result.TreeDepth+3)
		log.Printf("     - Decrease min_samples (current: 50 → try: 25)")
		log.Printf("     - Add more features")
	case "Low":
		log.Printf("   🟢 GOOD GENERALIZATION!")
		log.Printf("   Model appears well-balanced")
	}

	log.Printf("🎯 ==============================================================")
}

// SerializableNode represents a node that can be serialized to JSON
type SerializableNode struct {
	FeatureIndex int               `json:"feature_index"`
	Threshold    float64           `json:"threshold"`
	Left         *SerializableNode `json:"left"`
	Right        *SerializableNode `json:"right"`
	Value        float64           `json:"value"`
	IsLeaf       bool              `json:"is_leaf"`
	Samples      int               `json:"samples"`
	Gini         float64           `json:"gini"`
	ClassCounts  map[string]int    `json:"class_counts"` // Use string keys for JSON compatibility
}

// serializeNode converts a Node to a SerializableNode
func (c *CART) serializeNode(node *Node) *SerializableNode {
	if node == nil {
		return nil
	}

	// Convert ClassCounts from map[float64]int to map[string]int
	classCounts := make(map[string]int)
	if node.ClassCounts != nil {
		for class, count := range node.ClassCounts {
			classCounts[fmt.Sprintf("%.1f", class)] = count
		}
	}

	serialized := &SerializableNode{
		FeatureIndex: node.FeatureIndex,
		Threshold:    node.Threshold,
		Value:        node.Value,
		IsLeaf:       node.IsLeaf,
		Samples:      node.Samples,
		Gini:         node.Gini,
		ClassCounts:  classCounts,
	}

	// Recursively serialize children
	serialized.Left = c.serializeNode(node.Left)
	serialized.Right = c.serializeNode(node.Right)

	return serialized
}

// deserializeNode converts a SerializableNode back to a Node
func (c *CART) deserializeNode(serialized *SerializableNode) *Node {
	if serialized == nil {
		return nil
	}

	// Convert ClassCounts from map[string]int back to map[float64]int
	classCounts := make(map[float64]int)
	if serialized.ClassCounts != nil {
		for classStr, count := range serialized.ClassCounts {
			var classFloat float64
			if _, err := fmt.Sscanf(classStr, "%f", &classFloat); err == nil {
				classCounts[classFloat] = count
			}
		}
	}

	node := &Node{
		FeatureIndex: serialized.FeatureIndex,
		Threshold:    serialized.Threshold,
		Value:        serialized.Value,
		IsLeaf:       serialized.IsLeaf,
		Samples:      serialized.Samples,
		Gini:         serialized.Gini,
		ClassCounts:  classCounts,
	}

	// Recursively deserialize children
	node.Left = c.deserializeNode(serialized.Left)
	node.Right = c.deserializeNode(serialized.Right)

	return node
}

// GetNormalizationParams returns the stored normalization parameters
func (c *CART) GetNormalizationParams() *NormalizationParams {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if len(c.normalizationMins) == 0 || len(c.normalizationMaxs) == 0 {
		return nil
	}

	return &NormalizationParams{
		Mins: c.normalizationMins,
		Maxs: c.normalizationMaxs,
	}
}

// ApplyNormalization applies the stored normalization parameters to prediction features
func (c *CART) ApplyNormalization(features []float64) []float64 {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if len(c.normalizationMins) == 0 || len(c.normalizationMaxs) == 0 {
		log.Printf("⚠️ No normalization parameters stored, returning original features")
		return features
	}

	if len(features) != len(c.normalizationMins) {
		log.Printf("⚠️ Feature count mismatch for normalization: expected %d, got %d",
			len(c.normalizationMins), len(features))
		return features
	}

	normalized := make([]float64, len(features))
	featureNames := []string{"Area", "Sexo", "Edad", "Nivel educativo", "Categoría ocupacional", "Horas Trabajadas", "Ingreso mensual"}

	for i, feature := range features {
		min := c.normalizationMins[i]
		max := c.normalizationMaxs[i]

		if max > min {
			normalized[i] = (feature - min) / (max - min)
			// Clamp to [0,1] range
			if normalized[i] < 0 {
				normalized[i] = 0
			}
			if normalized[i] > 1 {
				normalized[i] = 1
			}
		} else {
			normalized[i] = 0.0 // All values in training were the same
		}

		// Log detailed normalization for debugging
		if i < len(featureNames) {
			log.Printf("  %s: %.2f -> %.4f (using min=%.2f, max=%.2f)",
				featureNames[i], feature, normalized[i], min, max)
		}
	}

	return normalized
}
