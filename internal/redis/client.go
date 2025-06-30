package redis

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// Client wraps Redis client with custom methods
type Client struct {
	rdb *redis.Client
	ctx context.Context
}

// NewClient creates a new Redis client
func NewClient() *Client {
	rdb := redis.NewClient(&redis.Options{
		Addr:     "redis:6379",
		Password: "", // no password
		DB:       0,  // default DB
	})

	ctx := context.Background()

	// Test connection
	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		fmt.Printf("Warning: Redis connection failed: %v\n", err)
		fmt.Println("Make sure Redis server is running on localhost:6379")
	}

	return &Client{
		rdb: rdb,
		ctx: ctx,
	}
}

// Set stores a key-value pair with optional expiration
func (c *Client) Set(key, value string) error {
	return c.rdb.Set(c.ctx, key, value, 24*time.Hour).Err()
}

// SetWithExpiration stores a key-value pair with custom expiration
func (c *Client) SetWithExpiration(key, value string, expiration time.Duration) error {
	return c.rdb.Set(c.ctx, key, value, expiration).Err()
}

// Get retrieves a value by key
func (c *Client) Get(key string) (string, error) {
	result, err := c.rdb.Get(c.ctx, key).Result()
	if err == redis.Nil {
		return "", fmt.Errorf("key not found: %s", key)
	}
	return result, err
}

// Delete removes a key
func (c *Client) Delete(key string) error {
	return c.rdb.Del(c.ctx, key).Err()
}

// Exists checks if a key exists
func (c *Client) Exists(key string) (bool, error) {
	result, err := c.rdb.Exists(c.ctx, key).Result()
	return result > 0, err
}

// SetHash stores a hash field
func (c *Client) SetHash(key, field, value string) error {
	return c.rdb.HSet(c.ctx, key, field, value).Err()
}

// GetHash retrieves a hash field
func (c *Client) GetHash(key, field string) (string, error) {
	result, err := c.rdb.HGet(c.ctx, key, field).Result()
	if err == redis.Nil {
		return "", fmt.Errorf("hash field not found: %s.%s", key, field)
	}
	return result, err
}

// GetAllHash retrieves all fields from a hash
func (c *Client) GetAllHash(key string) (map[string]string, error) {
	result, err := c.rdb.HGetAll(c.ctx, key).Result()
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Increment increments a counter
func (c *Client) Increment(key string) (int64, error) {
	return c.rdb.Incr(c.ctx, key).Result()
}

// IncrementBy increments a counter by a specific value
func (c *Client) IncrementBy(key string, value int64) (int64, error) {
	return c.rdb.IncrBy(c.ctx, key, value).Result()
}

// SetExpiration sets expiration for an existing key
func (c *Client) SetExpiration(key string, expiration time.Duration) error {
	return c.rdb.Expire(c.ctx, key, expiration).Err()
}

// GetTTL gets the time to live for a key
func (c *Client) GetTTL(key string) (time.Duration, error) {
	return c.rdb.TTL(c.ctx, key).Result()
}

// Keys returns all keys matching a pattern
func (c *Client) Keys(pattern string) ([]string, error) {
	return c.rdb.Keys(c.ctx, pattern).Result()
}

// FlushDB clears the current database
func (c *Client) FlushDB() error {
	return c.rdb.FlushDB(c.ctx).Err()
}

// Ping tests the connection
func (c *Client) Ping() error {
	return c.rdb.Ping(c.ctx).Err()
}

// Close closes the Redis connection
func (c *Client) Close() error {
	return c.rdb.Close()
}

// GetStats returns Redis statistics
func (c *Client) GetStats() (map[string]interface{}, error) {
	info, err := c.rdb.Info(c.ctx).Result()
	if err != nil {
		return nil, err
	}

	stats := map[string]interface{}{
		"info":      info,
		"connected": true,
	}

	// Get database size
	dbSize, err := c.rdb.DBSize(c.ctx).Result()
	if err == nil {
		stats["db_size"] = dbSize
	}

	return stats, nil
}

// StorePredictionResult stores a prediction result with metadata
func (c *Client) StorePredictionResult(id string, result map[string]interface{}) error {
	key := fmt.Sprintf("prediction:%s", id)

	// Store individual fields
	for field, value := range result {
		if err := c.SetHash(key, field, fmt.Sprintf("%v", value)); err != nil {
			return err
		}
	}

	// Set expiration for prediction results (24 hours)
	return c.SetExpiration(key, 24*time.Hour)
}

// GetPredictionResult retrieves a prediction result
func (c *Client) GetPredictionResult(id string) (map[string]string, error) {
	key := fmt.Sprintf("prediction:%s", id)
	return c.GetAllHash(key)
}

// StorePredictionStats stores prediction statistics
func (c *Client) StorePredictionStats(stats map[string]interface{}) error {
	key := "prediction_stats"

	for field, value := range stats {
		if err := c.SetHash(key, field, fmt.Sprintf("%v", value)); err != nil {
			return err
		}
	}

	return nil
}

// GetPredictionStats retrieves prediction statistics
func (c *Client) GetPredictionStats() (map[string]string, error) {
	return c.GetAllHash("prediction_stats")
}

// IncrementPredictionCounter increments the prediction counter
func (c *Client) IncrementPredictionCounter() (int64, error) {
	return c.Increment("prediction_count")
}

// SetModelMetadata stores model metadata
func (c *Client) SetModelMetadata(metadata map[string]interface{}) error {
	key := "model_metadata"

	for field, value := range metadata {
		if err := c.SetHash(key, field, fmt.Sprintf("%v", value)); err != nil {
			return err
		}
	}

	return nil
}

// GetModelMetadata retrieves model metadata
func (c *Client) GetModelMetadata() (map[string]string, error) {
	return c.GetAllHash("model_metadata")
}

// SetJSON stores a JSON value with expiration
func (c *Client) SetJSON(key string, value interface{}, expiration time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return c.rdb.Set(c.ctx, key, data, expiration).Err()
}

// GetJSON retrieves and unmarshals a JSON value
func (c *Client) GetJSON(key string, dest interface{}) error {
	data, err := c.rdb.Get(c.ctx, key).Result()
	if err != nil {
		return err
	}
	return json.Unmarshal([]byte(data), dest)
}

// SetModel stores a model with JSON serialization
func (c *Client) SetModel(key string, model interface{}) error {
	data, err := json.Marshal(model)
	if err != nil {
		return err
	}
	return c.rdb.Set(c.ctx, key, data, 24*time.Hour).Err()
}

// GetModel retrieves and deserializes a model
func (c *Client) GetModel(key string, model interface{}) error {
	data, err := c.rdb.Get(c.ctx, key).Result()
	if err != nil {
		return err
	}
	return json.Unmarshal([]byte(data), model)
}

// IncrementCounter increments a counter and returns the new value
func (c *Client) IncrementCounter(key string) error {
	return c.rdb.Incr(c.ctx, key).Err()
}
