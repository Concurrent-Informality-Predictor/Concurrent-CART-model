package data

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"go-ml-informalidad/internal/model"
)

// Loader handles data loading operations
type Loader struct {
	filePath string
}

// NewLoader creates a new data loader
func NewLoader(filePath string) *Loader {
	return &Loader{
		filePath: filePath,
	}
}

// LoadTrainingData loads training data from CSV file
func (l *Loader) LoadTrainingData() (*model.TrainingData, error) {
	file, err := os.Open(l.filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error reading CSV: %v", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("insufficient data in CSV file")
	}
	// Get headers
	headers := records[0]
	log.Printf("Dataset headers: %v", headers)

	// Map specific columns based on your dataset structure
	// Expected columns: Area, Sexo, Edad, Nivel educativo alcanzado, Categoría ocupacional, Horas Trabajadas, Ingreso mensual, Informalidad laboral
	featureColumns := map[string]int{
		"Area":                      -1,
		"Sexo":                      -1,
		"Edad":                      -1,
		"Nivel educativo alcanzado": -1,
		"Categoría ocupacional":     -1,
		"Horas Trabajadas":          -1,
		"Ingreso mensual":           -1,
	}
	targetColumn := -1

	// Find column indices
	for i, header := range headers {
		cleanHeader := strings.TrimSpace(header)
		switch {
		case strings.Contains(cleanHeader, "Area"):
			featureColumns["Area"] = i
		case strings.Contains(cleanHeader, "Sexo"):
			featureColumns["Sexo"] = i
		case strings.Contains(cleanHeader, "Edad"):
			featureColumns["Edad"] = i
		case strings.Contains(cleanHeader, "Nivel educativo"):
			featureColumns["Nivel educativo alcanzado"] = i
		case strings.Contains(cleanHeader, "Categor") && strings.Contains(cleanHeader, "ocupacional"):
			featureColumns["Categoría ocupacional"] = i
		case strings.Contains(cleanHeader, "Horas"):
			featureColumns["Horas Trabajadas"] = i
		case strings.Contains(cleanHeader, "Ingreso"):
			featureColumns["Ingreso mensual"] = i
		case strings.Contains(cleanHeader, "Informalidad"):
			targetColumn = i
		}
	}

	// Verify we found all required columns
	missingColumns := []string{}
	for colName, index := range featureColumns {
		if index == -1 {
			missingColumns = append(missingColumns, colName)
		}
	}
	if targetColumn == -1 {
		missingColumns = append(missingColumns, "Informalidad laboral")
	}

	if len(missingColumns) > 0 {
		return nil, fmt.Errorf("missing required columns: %v", missingColumns)
	}

	log.Printf("Found target column at index: %d", targetColumn)
	log.Printf("Feature columns mapping: %v", featureColumns)

	// Define ordered features for consistent feature vector
	orderedFeatures := []string{
		"Area",
		"Sexo",
		"Edad",
		"Nivel educativo alcanzado",
		"Categoría ocupacional",
		"Horas Trabajadas",
		"Ingreso mensual",
	}

	// Parse data with random sample limit for faster training
	var features [][]float64
	var labels []float64
	maxSamples := 600000 // Use 600,000 random samples for training

	// Create random indices for sampling
	totalRecords := len(records) - 1 // Exclude header
	var sampleIndices []int

	if totalRecords <= maxSamples {
		// Use all records if we have fewer than maxSamples
		sampleIndices = make([]int, totalRecords)
		for i := range sampleIndices {
			sampleIndices[i] = i + 1 // Start from 1 to skip header
		}
		log.Printf("🚀 Using all %d available records", totalRecords)
	} else {
		// Create random sample of maxSamples
		sampleIndices = l.createRandomSample(totalRecords, maxSamples)
		log.Printf("🚀 Using random sample of %d from %d total records", maxSamples, totalRecords)
	}

	processedSamples := 0
	for _, i := range sampleIndices {
		record := records[i]

		if len(record) <= targetColumn {
			continue // Skip incomplete rows
		}

		// Parse features in order
		featureRow := make([]float64, len(orderedFeatures))
		validRow := true

		for j, featureName := range orderedFeatures {
			idx := featureColumns[featureName]
			if idx >= len(record) {
				validRow = false
				break
			}

			value, err := l.parseFeature(record[idx], headers[idx])
			if err != nil {
				validRow = false
				break
			}
			featureRow[j] = value
		}

		if !validRow {
			continue // Skip invalid rows
		}

		// Parse label
		label, err := l.parseLabel(record[targetColumn])
		if err != nil {
			continue // Skip invalid rows
		}

		features = append(features, featureRow)
		labels = append(labels, label)
		processedSamples++

		// Log progress every 1000 samples
		if processedSamples%1000 == 0 {
			log.Printf("📊 Loaded %d/%d samples...", processedSamples, maxSamples)
		}
	}

	if len(features) == 0 {
		return nil, fmt.Errorf("no valid data found")
	}

	// Normalize features and store normalization parameters
	var normParams *model.NormalizationParams
	features, normParams = l.normalizeFeatures(features)

	metadata := map[string]interface{}{
		"num_samples":          len(features),
		"num_features":         len(features[0]),
		"headers":              orderedFeatures,
		"target":               headers[targetColumn],
		"normalization_params": normParams,
	}

	return &model.TrainingData{
		Features: features,
		Labels:   labels,
		Metadata: metadata,
	}, nil
}

// parseFeature parses a feature value from string
func (l *Loader) parseFeature(value, header string) (float64, error) {
	value = strings.TrimSpace(value)

	// Handle empty values first
	if value == "" || value == "NA" || value == "NULL" {
		return 0.0, nil // Handle missing values as 0
	}

	// Handle categorical variables
	if l.isCategorical(header, value) {
		return l.encodeCategorical(header, value), nil
	}

	// Handle numeric variables
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		// If it fails to parse as numeric but header suggests it should be numeric
		log.Printf("⚠️ Failed to parse numeric value '%s' for header '%s', using 0.0", value, header)
		return 0.0, nil // Use 0 instead of error for robustness
	}

	return parsed, nil
}

// parseLabel parses the target label
func (l *Loader) parseLabel(value string) (float64, error) {
	value = strings.TrimSpace(value)
	originalValue := value
	value = strings.ToLower(value)

	// Handle specific values from your dataset: ['Informal', 'Formal']
	switch value {
	case "formal":
		return 1.0, nil // Formal employment
	case "informal":
		return 0.0, nil // Informal employment
	// Fallback for other possible formats
	case "1", "true", "yes", "si":
		return 1.0, nil
	case "0", "false", "no":
		return 0.0, nil
	default:
		// Try to parse as number
		parsed, err := strconv.ParseFloat(originalValue, 64)
		if err != nil {
			log.Printf("⚠️ Unknown label value: '%s'", originalValue)
			return 0.0, fmt.Errorf("error parsing label: %v", err)
		}

		// Convert to binary (0 or 1)
		if parsed > 0.5 {
			return 1.0, nil
		}
		return 0.0, nil
	}
}

// isCategorical determines if a field is categorical
func (l *Loader) isCategorical(header, value string) bool {
	header = strings.ToLower(header)

	// First, check if the header is definitely numeric
	numericHeaders := []string{"edad", "horas", "ingreso"}
	for _, numericHeader := range numericHeaders {
		if strings.Contains(header, numericHeader) {
			return false // These are always numeric
		}
	}

	// Check if header suggests categorical data
	categoricalHeaders := []string{"sexo", "genero", "nivel", "educativo", "area", "residencia", "categoria", "ocupacional"}
	for _, cat := range categoricalHeaders {
		if strings.Contains(header, cat) {
			return true
		}
	}

	// Check if value is obviously categorical (non-numeric)
	_, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	return err != nil
}

// encodeCategorical encodes categorical variables to numeric
func (l *Loader) encodeCategorical(header, value string) float64 {
	header = strings.ToLower(header)
	originalValue := value
	value = strings.ToLower(strings.TrimSpace(value))

	// Area encoding: ['Urbano', 'Rural']
	if strings.Contains(header, "area") || strings.Contains(header, "residencia") {
		switch value {
		case "urbano":
			return 1.0
		case "rural":
			return 0.0
		default:
			log.Printf("⚠️ Unknown Area value: '%s'", originalValue)
			return 0.5 // Unknown
		}
	}

	// Gender encoding: ['Hombre', 'Mujer']
	if strings.Contains(header, "sexo") {
		switch value {
		case "hombre":
			return 1.0
		case "mujer":
			return 0.0
		default:
			log.Printf("⚠️ Unknown Sexo value: '%s'", originalValue)
			return 0.5 // Unknown
		}
	}

	// Education level encoding (ordered by education level)
	// ['Sin nivel', 'Educación Inicial', 'Básica especial', 'Primaria incompleta',
	//  'Primaria completa', 'Secundaria incompleta', 'Secundaria completa',
	//  'Superior no universitaria incompleta', 'Superior no universitaria completa',
	//  'Superior universitaria incompleta', 'Superior universitaria completa', 'Maestría/Doctorado']
	if strings.Contains(header, "educativo") || strings.Contains(header, "nivel") {
		// Comprehensive normalization to handle all encoding issues
		normalizedValue := l.cleanEncodingIssues(value)

		switch normalizedValue {
		case "sin nivel":
			return 0.0
		case "educación inicial":
			return 1.0
		case "básica especial":
			return 2.0
		case "primaria incompleta":
			return 3.0
		case "primaria completa":
			return 4.0
		case "secundaria incompleta":
			return 5.0
		case "secundaria completa":
			return 6.0
		case "superior no universitaria incompleta":
			return 7.0
		case "superior no universitaria completa":
			return 8.0
		case "superior universitaria incompleta":
			return 9.0
		case "superior universitaria completa":
			return 10.0
		case "maestría/doctorado":
			return 11.0
		default:
			log.Printf("⚠️ Unknown Nivel educativo value: '%s' (original: '%s')", normalizedValue, originalValue)
			return 4.0 // Default to primaria completa
		}
	}

	// Occupational category encoding
	// ['Ayudante en un negocio de la familia', 'Empleado u obrero', 'Trabajador independiente',
	//  'Empleador o patrono', 'Trabajador del hogar', 'Ayudante en un negocio de la familia de otro hogar',
	//  'Aprendiz/practicante remunerado', 'Practicante no remunerado']
	if strings.Contains(header, "categoria") || strings.Contains(header, "ocupacional") {
		switch value {
		case "ayudante en un negocio de la familia":
			return 0.0
		case "practicante no remunerado":
			return 1.0
		case "aprendiz/practicante remunerado":
			return 2.0
		case "trabajador del hogar":
			return 3.0
		case "ayudante en un negocio de la familia de otro hogar":
			return 4.0
		case "empleado u obrero":
			return 5.0
		case "trabajador independiente":
			return 6.0
		case "empleador o patrono":
			return 7.0
		default:
			log.Printf("⚠️ Unknown Categoría ocupacional value: '%s'", originalValue)
			return 5.0 // Default to empleado u obrero
		}
	}

	// Default: simple hash-based encoding for unknown categories
	// But only log if it's actually supposed to be categorical
	if l.isCategorical(header, originalValue) {
		log.Printf("⚠️ Unknown categorical field: header='%s', value='%s'", header, originalValue)
		hash := 0
		for _, char := range value {
			hash = (hash*31 + int(char)) % 10
		}
		return float64(hash) / 10.0
	}

	// If we reach here, something went wrong - this shouldn't be categorical
	return 0.0
}

// normalizeFeatures normalizes feature values using min-max normalization
func (l *Loader) normalizeFeatures(features [][]float64) ([][]float64, *model.NormalizationParams) {
	if len(features) == 0 {
		return features, nil
	}

	numFeatures := len(features[0])
	mins := make([]float64, numFeatures)
	maxs := make([]float64, numFeatures)

	// Initialize mins and maxs
	for j := 0; j < numFeatures; j++ {
		mins[j] = features[0][j]
		maxs[j] = features[0][j]
	}

	// Find min and max for each feature
	for i := 0; i < len(features); i++ {
		for j := 0; j < numFeatures; j++ {
			if features[i][j] < mins[j] {
				mins[j] = features[i][j]
			}
			if features[i][j] > maxs[j] {
				maxs[j] = features[i][j]
			}
		}
	}

	// Log normalization parameters for debugging
	featureNames := []string{"Area", "Sexo", "Edad", "Nivel educativo", "Categoría ocupacional", "Horas Trabajadas", "Ingreso mensual"}
	log.Printf("🔢 Normalization parameters:")
	for j := 0; j < numFeatures && j < len(featureNames); j++ {
		log.Printf("  %s: min=%.2f, max=%.2f", featureNames[j], mins[j], maxs[j])
	}

	// Normalize features
	normalized := make([][]float64, len(features))
	for i := 0; i < len(features); i++ {
		normalized[i] = make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			if maxs[j] > mins[j] {
				normalized[i][j] = (features[i][j] - mins[j]) / (maxs[j] - mins[j])
			} else {
				normalized[i][j] = 0.0 // All values are the same
			}
		}
	}

	// Create normalization parameters object
	params := &model.NormalizationParams{
		Mins: mins,
		Maxs: maxs,
	}

	return normalized, params
}

// LoadPredictionData loads data for prediction (single row) using stored normalization parameters
func (l *Loader) LoadPredictionData(data map[string]interface{}) ([]float64, error) {
	// Expected features must match training data order
	orderedFeatures := []string{
		"Area",
		"Sexo",
		"Edad",
		"Nivel educativo alcanzado",
		"Categoría ocupacional",
		"Horas Trabajadas",
		"Ingreso mensual",
	}

	// Map input keys to expected feature names
	keyMapping := map[string]string{
		"area":                  "Area",
		"sexo":                  "Sexo",
		"edad":                  "Edad",
		"nivel_educativo":       "Nivel educativo alcanzado",
		"categoria_ocupacional": "Categoría ocupacional",
		"horas_trabajadas":      "Horas Trabajadas",
		"ingreso_mensual":       "Ingreso mensual",
	}

	features := make([]float64, len(orderedFeatures))

	for i, featureName := range orderedFeatures {
		// Find the input key for this feature
		var inputKey string
		var value interface{}
		var exists bool

		// Try to find the value using the key mapping
		for key, mappedFeature := range keyMapping {
			if mappedFeature == featureName {
				inputKey = key
				value, exists = data[key]
				break
			}
		}

		if !exists {
			return nil, fmt.Errorf("missing feature: %s (looking for key: %s)", featureName, inputKey)
		}

		// Convert to float64
		var floatValue float64
		switch v := value.(type) {
		case float64:
			floatValue = v
		case int:
			floatValue = float64(v)
		case string:
			// Use the same parsing logic as training data
			parsed, err := l.parseFeature(v, featureName)
			if err != nil {
				return nil, fmt.Errorf("error parsing feature %s: %v", featureName, err)
			}
			floatValue = parsed
		default:
			return nil, fmt.Errorf("unsupported type for feature %s", featureName)
		}

		features[i] = floatValue
	}

	// Log raw features before normalization for debugging
	log.Printf("🔍 Raw prediction features: %v", features)

	// Note: Normalization will be applied in the prediction handler using stored parameters
	// This method now returns raw (unnormalized) features
	return features, nil
}

// LoadPredictionDataWithNormalization loads and normalizes prediction data using stored parameters
func (l *Loader) LoadPredictionDataWithNormalization(data map[string]interface{}, normParams *model.NormalizationParams) ([]float64, error) {
	// Get raw features
	features, err := l.LoadPredictionData(data)
	if err != nil {
		return nil, err
	}

	// Apply normalization using stored parameters
	if normParams != nil {
		features = l.applyNormalization(features, normParams)
		log.Printf("🔍 Normalized prediction features: %v", features)
	}

	return features, nil
}

// applyNormalization applies stored normalization parameters to features
func (l *Loader) applyNormalization(features []float64, params *model.NormalizationParams) []float64 {
	if params == nil || len(params.Mins) != len(features) || len(params.Maxs) != len(features) {
		log.Printf("⚠️ Invalid normalization parameters, using fallback normalization")
		return l.normalizeSingleSample(features)
	}

	normalized := make([]float64, len(features))
	featureNames := []string{"Area", "Sexo", "Edad", "Nivel educativo", "Categoría ocupacional", "Horas Trabajadas", "Ingreso mensual"}

	for i, feature := range features {
		min := params.Mins[i]
		max := params.Maxs[i]

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

// normalizeSingleSample normalizes a single sample (for prediction)
// This is a simplified version - in production you should store training normalization params
func (l *Loader) normalizeSingleSample(features []float64) []float64 {
	// For now, apply basic normalization
	// TODO: Store min/max values from training and reuse them here
	normalized := make([]float64, len(features))

	// Expected ranges based on the actual categorical encodings
	expectedRanges := []struct{ min, max float64 }{
		{0.0, 1.0},     // Area (0=Rural, 1=Urbano)
		{0.0, 1.0},     // Sexo (0=Mujer, 1=Hombre)
		{18.0, 80.0},   // Edad (numeric)
		{0.0, 11.0},    // Nivel educativo (0=Sin nivel, 11=Maestría/Doctorado)
		{0.0, 7.0},     // Categoría ocupacional (0=Ayudante familia, 7=Empleador)
		{0.0, 60.0},    // Horas trabajadas (numeric)
		{0.0, 10000.0}, // Ingreso mensual (numeric)
	}

	for i, feature := range features {
		if i < len(expectedRanges) {
			min := expectedRanges[i].min
			max := expectedRanges[i].max
			if max > min {
				normalized[i] = (feature - min) / (max - min)
				// Clamp to [0,1]
				if normalized[i] < 0 {
					normalized[i] = 0
				}
				if normalized[i] > 1 {
					normalized[i] = 1
				}
			} else {
				normalized[i] = 0
			}
		} else {
			normalized[i] = feature
		}
	}

	return normalized
}

// createRandomSample generates random sample indices for data sampling
func (l *Loader) createRandomSample(totalRecords, sampleSize int) []int {
	if sampleSize >= totalRecords {
		// Return all indices if sample size is larger than total
		indices := make([]int, totalRecords)
		for i := range indices {
			indices[i] = i + 1 // Start from 1 to skip header
		}
		return indices
	}

	// Create all possible indices
	allIndices := make([]int, totalRecords)
	for i := range allIndices {
		allIndices[i] = i + 1 // Start from 1 to skip header
	}

	// Simple shuffle using current time as seed for randomization
	seed := time.Now().UnixNano()
	for i := len(allIndices) - 1; i > 0; i-- {
		seed = seed*1103515245 + 12345 // Linear congruential generator
		j := int(seed) % (i + 1)
		if j < 0 {
			j = -j
		}
		allIndices[i], allIndices[j] = allIndices[j], allIndices[i]
	}

	// Return the first sampleSize indices
	return allIndices[:sampleSize]
}

// cleanEncodingIssues handles encoding problems commonly found in CSV files with Spanish text
func (l *Loader) cleanEncodingIssues(value string) string {
	// Normalize to lower case and trim spaces
	normalized := strings.ToLower(strings.TrimSpace(value))

	// Handle common UTF-8 encoding issues where special characters get corrupted
	replacements := map[string]string{
		// Tildes/accents issues
		"ã¡":      "á", // á corrupted as ã¡
		"ã©":      "é", // é corrupted as ã©
		"ã\u00ad": "í", // í corrupted with soft hyphen
		"ã³":      "ó", // ó corrupted as ã³
		"ãº":      "ú", // ú corrupted as ãº
		"ã±":      "ñ", // ñ corrupted as ã±

		// Alternative corruption patterns
		"ã": "ó", // Often ó gets corrupted as just ã
		"�": "í", // Question mark replacement for í
		"â": "ó", // Another pattern for ó

		// Handle Windows-1252 to UTF-8 conversion issues
		"educaciã³n": "educación",
		"educaci�n":  "educación",
		"educaciòn":  "educación",
		"educaciÃ³n": "educación",

		"bãsica":  "básica",
		"bÃ¡sica": "básica",
		"b�sica":  "básica",

		"primaria":   "primaria",   // This one is usually fine
		"secundaria": "secundaria", // This one is usually fine

		"superior":      "superior",      // This one is usually fine
		"universitaria": "universitaria", // This one is usually fine

		"maestrãa":      "maestría",
		"maestr\u00ada": "maestría",
		"maestr�a":      "maestría",
		"maestrÃ\u00ad": "maestría",

		// Fix specific education values we've seen in logs
		"educaciã³n inicial": "educación inicial",
		"educaci�n inicial":  "educación inicial",
		"bãsica especial":    "básica especial",
		"b�sica especial":    "básica especial",
	}

	// Apply all replacements
	for corrupted, correct := range replacements {
		normalized = strings.ReplaceAll(normalized, corrupted, correct)
	}

	// Additional cleanup for any remaining encoding artifacts
	// Remove zero-width characters and other invisible unicode
	normalized = strings.ReplaceAll(normalized, "\u200b", "")  // Zero-width space
	normalized = strings.ReplaceAll(normalized, "\ufeff", "")  // BOM
	normalized = strings.ReplaceAll(normalized, "\u00a0", " ") // Non-breaking space to regular space

	// Final trim
	normalized = strings.TrimSpace(normalized)

	return normalized
}
