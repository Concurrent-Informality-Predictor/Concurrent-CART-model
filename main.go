package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"go-ml-informalidad/internal/model"
	"go-ml-informalidad/internal/redis"
	"go-ml-informalidad/internal/server"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize Redis client
	redisClient := redis.NewClient()
	defer redisClient.Close()

	// Initialize ML model (CART)
	mlModel := model.NewCART()

	// Try to load existing model from Redis
	log.Printf("🔍 Attempting to load existing model from Redis...")
	if err := mlModel.LoadModelFromRedis(redisClient); err != nil {
		log.Printf("⚠️ No existing model found in Redis: %v", err)
		log.Printf("💡 Use /api/model/train endpoint to train a new model")
	} else {
		log.Printf("✅ Successfully loaded existing model from Redis")
	}

	// Create channels for communication
	trainingChan := make(chan *model.TrainingData, 100)
	predictionChan := make(chan *model.PredictionRequest, 100)
	resultChan := make(chan *model.PredictionResult, 100)

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		mlModel.StartTrainingWorker(ctx, trainingChan, redisClient)
	}()

	numPredictionWorkers := 3
	for i := 0; i < numPredictionWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			mlModel.StartPredictionWorker(ctx, workerID, predictionChan, resultChan, redisClient)
		}(i)
	}

	httpServer := server.NewServer(predictionChan, resultChan, mlModel)

	// Start HTTP server in goroutine
	go func() {
		port := os.Getenv("PORT")
		if port == "" {
			port = "8081" // Puerto por defecto
		}
		log.Printf("Starting HTTP server on :%s", port)
		if err := http.ListenAndServe(":" + port, httpServer.Router()); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	<-sigChan
	log.Println("Shutting down gracefully...")

	cancel()

	// Give goroutines time to finish
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		log.Println("All goroutines finished")
	case <-time.After(10 * time.Second):
		log.Println("Timeout waiting for goroutines to finish")
	}

	fmt.Println("Application terminated")
}
