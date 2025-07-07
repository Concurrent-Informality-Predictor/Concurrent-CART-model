package gateway

import (
    "net/http"
    "time"
    "github.com/gorilla/mux"
)

type Gateway struct {
    nodes   []string
    counter int 
}

// NewGateway crea una nueva instancia del Gateway
func NewGateway(nodes []string) *Gateway {
    return &Gateway{
        nodes:   nodes,
        counter: 0,
    }
}

func (g *Gateway) getNextNode() string {
    node := g.nodes[g.counter%len(g.nodes)]
    g.counter++
    return node
}

func (g *Gateway) Router() *mux.Router {
    r := mux.NewRouter()
    r.HandleFunc("/api/predict", g.handlePredict).Methods("POST", "OPTIONS")
    r.HandleFunc("/api/predict/batch", g.handlePredictBatch).Methods("POST", "OPTIONS")
    r.HandleFunc("/api/model/train", g.handleTrain).Methods("POST", "OPTIONS")
    r.HandleFunc("/api/health", g.handleHealth).Methods("GET")
    return r
}

func (g *Gateway) handleHealth(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    w.Write([]byte(`{"status": "gateway running", "timestamp": "` + time.Now().Format(time.RFC3339) + `"}`))
}