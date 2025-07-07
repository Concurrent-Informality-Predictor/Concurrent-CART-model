package main

import (
    "log"
    "net/http"
    "os"
    "strings"

    "go-ml-informalidad/internal/gateway"
)

func main() {
    // Lee nodos desde variable de entorno o hardcodea para pruebas
    nodesStr := os.Getenv("NODES")
    var nodes []string
    if nodesStr == "" {
        // Por defecto: para docker compose local
        nodes = []string{
            "http://nodo1:8081",
            "http://nodo2:8082",
            "http://nodo3:8083",
        }
    } else {
        nodes = strings.Split(nodesStr, ",")
    }

    gw := gateway.NewGateway(nodes)

    port := os.Getenv("GATEWAY_PORT")
    if port == "" {
        port = "8000"
    }
    log.Printf("Gateway escuchando en :%s", port)
    log.Printf("Nodos backend: %v", nodes)
    http.ListenAndServe(":"+port, gw.Router())
}