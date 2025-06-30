package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"go-ml-informalidad/internal/data"
	"go-ml-informalidad/internal/model"
	"go-ml-informalidad/internal/redis"

	"github.com/gorilla/mux"
)

// Server handles HTTP requests
type Server struct {
	predictionChan chan<- *model.PredictionRequest
	resultChan     <-chan *model.PredictionResult
	dataLoader     *data.Loader
	pendingResults map[string]chan *model.PredictionResult
	cartModel      *model.CART
}

// NewServer creates a new HTTP server
func NewServer(predictionChan chan<- *model.PredictionRequest, resultChan <-chan *model.PredictionResult, cartModel *model.CART) *Server {
	return &Server{
		predictionChan: predictionChan,
		resultChan:     resultChan,
		dataLoader:     data.NewLoader(""),
		pendingResults: make(map[string]chan *model.PredictionResult),
		cartModel:      cartModel,
	}
}

// Router returns the HTTP router
func (s *Server) Router() *mux.Router {
	r := mux.NewRouter()

	// API routes
	r.HandleFunc("/api/predict", s.handlePredict).Methods("POST", "OPTIONS")
	r.HandleFunc("/api/health", s.handleHealth).Methods("GET")
	r.HandleFunc("/api/model/info", s.handleModelInfo).Methods("GET")
	r.HandleFunc("/api/model/train", s.handleModelTrain).Methods("POST", "OPTIONS")
	r.HandleFunc("/api/model/validate", s.handleModelValidation).Methods("POST", "OPTIONS")

	// Start result processor
	go s.processResults()

	return r
}

// PredictionRequest represents the request body for predictions
type PredictionRequest struct {
	Area           string  `json:"area"`
	Sexo           string  `json:"sexo"`
	Edad           float64 `json:"edad"`
	NivelEducativo string  `json:"nivel_educativo"`
	Categoria      string  `json:"categoria_ocupacional"`
	Horas          float64 `json:"horas_trabajadas"`
	Ingreso        float64 `json:"ingreso_mensual"`
}

// handlePredict handles prediction requests
func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	// Handle CORS preflight
	if r.Method == "OPTIONS" {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	var req PredictionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error": "Invalid JSON"}`, http.StatusBadRequest)
		return
	}

	// Create input data map to use the same processing as training
	inputData := map[string]interface{}{
		"area":                  req.Area,
		"sexo":                  req.Sexo,
		"edad":                  req.Edad,
		"nivel_educativo":       req.NivelEducativo,
		"categoria_ocupacional": req.Categoria,
		"horas_trabajadas":      req.Horas,
		"ingreso_mensual":       req.Ingreso,
	}

	// Use the SAME processing as training data
	loader := data.NewLoader("")
	features, err := loader.LoadPredictionData(inputData)
	if err != nil {
		log.Printf("❌ Error processing features: %v", err)
		http.Error(w, fmt.Sprintf(`{"error": "Error processing features: %v"}`, err), http.StatusBadRequest)
		return
	}

	log.Printf("🔍 Raw prediction features: %v", features)

	// Apply the SAME normalization as training data using stored parameters
	normalizedFeatures := s.cartModel.ApplyNormalization(features)
	log.Printf("🔍 Normalized prediction features: %v", normalizedFeatures)

	// Create prediction request with properly processed features
	predReq := &model.PredictionRequest{
		ID:        fmt.Sprintf("pred_%d", time.Now().UnixNano()),
		Features:  normalizedFeatures, // Now uses the SAME normalization as training
		Timestamp: time.Now(),
	}

	// Create result channel for this request
	resultChan := make(chan *model.PredictionResult, 1)
	s.pendingResults[predReq.ID] = resultChan

	// Send prediction request
	select {
	case s.predictionChan <- predReq:
		log.Printf("Prediction request sent: %s", predReq.ID)
	default:
		delete(s.pendingResults, predReq.ID)
		http.Error(w, `{"error": "Server busy, try again later"}`, http.StatusServiceUnavailable)
		return
	}

	// Wait for result with timeout
	select {
	case result := <-resultChan:
		delete(s.pendingResults, predReq.ID)
		log.Printf("✅ Prediction result: Class=%d, Probability=%.4f, Confidence=%.4f",
			result.Class, result.Probability, result.Confidence)
		json.NewEncoder(w).Encode(result)
	case <-time.After(30 * time.Second):
		delete(s.pendingResults, predReq.ID)
		http.Error(w, `{"error": "Prediction timeout"}`, http.StatusRequestTimeout)
	}
}

// handleModelInfo returns model information
func (s *Server) handleModelInfo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Get real model information from CART
	modelInfo := s.cartModel.GetModelInfo()

	// Add additional server information
	info := map[string]interface{}{
		"model_type":  modelInfo["model_type"],
		"trained":     modelInfo["trained"],
		"max_depth":   modelInfo["max_depth"],
		"min_samples": modelInfo["min_samples"],
		"features":    []string{"Area", "Sexo", "Edad", "Nivel educativo alcanzado", "Categoría ocupacional", "Horas Trabajadas", "Ingreso mensual"},
		"target":      "Informalidad laboral",
		"description": "Modelo predictivo de informalidad laboral basado en árboles de decisión CART",
		"timestamp":   time.Now(),
	}

	// Add tree-specific info if model is trained
	if trained, ok := modelInfo["trained"].(bool); ok && trained {
		if treeDepth, exists := modelInfo["tree_depth"]; exists {
			info["tree_depth"] = treeDepth
		}
		if leafCount, exists := modelInfo["leaf_count"]; exists {
			info["leaf_count"] = leafCount
		}
		if minImpurity, exists := modelInfo["min_impurity"]; exists {
			info["min_impurity"] = minImpurity
		}
	}

	json.NewEncoder(w).Encode(info)
}

// handleHealth returns server health status
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	health := map[string]interface{}{
		"status":     "healthy",
		"timestamp":  time.Now(),
		"version":    "1.0.0",
		"model_type": "CART Decision Tree",
	}

	json.NewEncoder(w).Encode(health)
}

// processResults processes prediction results from the result channel
func (s *Server) processResults() {
	for result := range s.resultChan {
		if resultChan, exists := s.pendingResults[result.ID]; exists {
			select {
			case resultChan <- result:
				log.Printf("Result delivered for prediction: %s", result.ID)
			default:
				log.Printf("Result channel full for prediction: %s", result.ID)
			}
		} else {
			log.Printf("No pending result channel for prediction: %s", result.ID)
		}
	}
}

// handleModelValidation handles model validation requests (validation only, no training)
func (s *Server) handleModelValidation(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Check if model is trained
	modelInfo := s.cartModel.GetModelInfo()
	if trained, ok := modelInfo["trained"].(bool); !ok || !trained {
		http.Error(w, `{"error": "Model not trained. Use /api/model/train endpoint first"}`, http.StatusBadRequest)
		return
	}

	// Parse request body for validation parameters
	var req struct {
		TestRatio float64 `json:"test_ratio,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// If no body or invalid JSON, use default test ratio
		req.TestRatio = 0.2 // Default 20% for testing
	}

	// Validate test ratio
	if req.TestRatio <= 0 || req.TestRatio >= 1 {
		req.TestRatio = 0.2 // Default to 20%
	}

	log.Printf("🔍 Starting model validation with test ratio: %.1f%%", req.TestRatio*100)

	// Load fresh data for validation (not training)
	loader := data.NewLoader("consolidate_clean_data.csv")
	validationData, err := loader.LoadTrainingData()
	if err != nil {
		log.Printf("❌ Error loading validation data: %v", err)
		http.Error(w, fmt.Sprintf("Error loading validation data: %v", err), http.StatusInternalServerError)
		return
	}

	log.Printf("📊 Loaded %d samples for validation", len(validationData.Features))

	// Split data for validation (don't train, just evaluate the existing model)
	totalSamples := len(validationData.Features)
	testSize := int(float64(totalSamples) * req.TestRatio)

	// Create random indices for test data
	testIndices := make([]int, testSize)
	for i := 0; i < testSize; i++ {
		testIndices[i] = i
	}

	// Extract test data
	testFeatures := make([][]float64, testSize)
	testLabels := make([]float64, testSize)
	for i, idx := range testIndices {
		testFeatures[i] = validationData.Features[idx]
		testLabels[i] = validationData.Labels[idx]
	}

	log.Printf("🎯 Evaluating existing model on %d test samples", len(testFeatures))

	// Evaluate the EXISTING trained model (no training)
	accuracy, predictions := s.evaluateModel(testFeatures, testLabels)

	// Calculate detailed metrics
	metrics := s.calculateValidationMetrics(testLabels, predictions)

	// Create validation result
	result := map[string]interface{}{
		"validation_type":  "existing_model_evaluation",
		"test_samples":     len(testFeatures),
		"test_accuracy":    accuracy,
		"test_precision":   metrics["precision"],
		"test_recall":      metrics["recall"],
		"test_f1_score":    metrics["f1_score"],
		"confusion_matrix": metrics["confusion_matrix"],
		"model_info":       s.cartModel.GetModelInfo(),
		"timestamp":        time.Now(),
	}

	log.Printf("✅ Model validation completed successfully")

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// handleModelTrain handles model training requests (separated from validation)
func (s *Server) handleModelTrain(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse request body for training parameters
	var req struct {
		ForceRetrain bool `json:"force_retrain,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// If no body or invalid JSON, continue with default values
		req.ForceRetrain = false
	}

	log.Printf("🚀 Starting model training (force_retrain: %v)", req.ForceRetrain)

	// Check if model is already trained
	if !req.ForceRetrain {
		modelInfo := s.cartModel.GetModelInfo()
		if trained, ok := modelInfo["trained"].(bool); ok && trained {
			log.Printf("⚠️ Model already trained. Use force_retrain=true to retrain")
			response := map[string]interface{}{
				"status":     "already_trained",
				"message":    "Model is already trained. Use force_retrain=true to retrain",
				"model_info": modelInfo,
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
			return
		}
	}

	// Load training data
	loader := data.NewLoader("consolidate_clean_data.csv")
	trainingData, err := loader.LoadTrainingData()
	if err != nil {
		log.Printf("❌ Error loading training data: %v", err)
		http.Error(w, fmt.Sprintf("Error loading training data: %v", err), http.StatusInternalServerError)
		return
	}

	log.Printf("📊 Loaded %d samples for training", len(trainingData.Features))

	// Train the model with normalization parameters
	startTime := time.Now()
	err = s.cartModel.TrainWithData(trainingData)
	if err != nil {
		log.Printf("❌ Training error: %v", err)
		http.Error(w, fmt.Sprintf("Training error: %v", err), http.StatusInternalServerError)
		return
	}

	trainingTime := time.Since(startTime)
	log.Printf("✅ Model training completed in %v", trainingTime)

	// Save model to Redis
	redisClient := redis.NewClient()
	defer redisClient.Close()

	if err := s.cartModel.SaveModelToRedis(redisClient); err != nil {
		log.Printf("⚠️ Warning: Could not save model to Redis: %v", err)
	} else {
		log.Printf("💾 Model saved to Redis successfully")
	}

	// Return training results
	response := map[string]interface{}{
		"status":        "training_completed",
		"samples":       len(trainingData.Features),
		"features":      len(trainingData.Features[0]),
		"training_time": trainingTime.String(),
		"model_info":    s.cartModel.GetModelInfo(),
		"timestamp":     time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// evaluateModel evaluates the existing model on test data
func (s *Server) evaluateModel(testFeatures [][]float64, testLabels []float64) (float64, []int) {
	correct := 0
	predictions := make([]int, len(testFeatures))

	for i := 0; i < len(testFeatures); i++ {
		prediction, err := s.cartModel.Predict(testFeatures[i])
		if err != nil {
			log.Printf("⚠️ Prediction error for sample %d: %v", i, err)
			predictions[i] = 0 // Default to informal
			continue
		}

		predictions[i] = prediction.Class
		actual := int(testLabels[i])

		if predictions[i] == actual {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(testFeatures))
	return accuracy, predictions
}

// calculateValidationMetrics calculates precision, recall, F1-score and confusion matrix
func (s *Server) calculateValidationMetrics(testLabels []float64, predictions []int) map[string]interface{} {
	tp, fp, tn, fn := 0, 0, 0, 0

	for i := 0; i < len(testLabels); i++ {
		predicted := predictions[i]
		actual := int(testLabels[i])

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

	return map[string]interface{}{
		"precision": precision,
		"recall":    recall,
		"f1_score":  f1Score,
		"confusion_matrix": map[string]int{
			"true_positive":  tp,
			"false_positive": fp,
			"true_negative":  tn,
			"false_negative": fn,
		},
	}
}
