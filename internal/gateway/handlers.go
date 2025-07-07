package gateway

import (
    "bytes"
    "io"
    "log"
    "net/http"
    "sync"
    "encoding/json"
)

func (g *Gateway) handlePredict(w http.ResponseWriter, r *http.Request) {
    targetNode := g.getNextNode()
    predictURL := targetNode + "/api/predict"

    body, err := io.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Error leyendo el body", http.StatusBadRequest)
        return
    }
    defer r.Body.Close()

    req, err := http.NewRequest("POST", predictURL, bytes.NewReader(body))
    if err != nil {
        http.Error(w, "Error creando la request al nodo", http.StatusInternalServerError)
        return
    }
    req.Header = r.Header

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        http.Error(w, "Error contactando al nodo", http.StatusBadGateway)
        return
    }
    defer resp.Body.Close()

    w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
    w.WriteHeader(resp.StatusCode)
    io.Copy(w, resp.Body)
    log.Printf("Request /api/predict enviada a %s", predictURL)
}

func (g *Gateway) handleTrain(w http.ResponseWriter, r *http.Request) {
    targetNode := g.nodes[0]
    trainURL := targetNode + "/api/model/train"

    body, err := io.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Error leyendo el body", http.StatusBadRequest)
        return
    }
    defer r.Body.Close()

    req, err := http.NewRequest("POST", trainURL, bytes.NewReader(body))
    if err != nil {
        http.Error(w, "Error creando la request al nodo", http.StatusInternalServerError)
        return
    }
    req.Header = r.Header

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        http.Error(w, "Error contactando al nodo backend", http.StatusBadGateway)
        return
    }
    defer resp.Body.Close()

    w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
    w.WriteHeader(resp.StatusCode)
    io.Copy(w, resp.Body)
}

func (g *Gateway) handlePredictBatch(w http.ResponseWriter, r *http.Request) {
    var inputs []map[string]interface{}
    err := json.NewDecoder(r.Body).Decode(&inputs)
    if err != nil {
        http.Error(w, "JSON inválido", http.StatusBadRequest)
        return
    }
    defer r.Body.Close()

    n := len(inputs)
    numNodes := len(g.nodes)
    if n == 0 || numNodes == 0 {
        http.Error(w, "No hay inputs o nodos", http.StatusBadRequest)
        return
    }

    groups := make([][]map[string]interface{}, numNodes)
    for i, input := range inputs {
        groups[i%numNodes] = append(groups[i%numNodes], input)
    }

    results := make([]interface{}, n)
    errs := make([]error, numNodes)
    var wg sync.WaitGroup
    wg.Add(numNodes)

    for idx, group := range groups {
        go func(idx int, group []map[string]interface{}) {
            defer wg.Done()
            if len(group) == 0 {
                return
            }
            for j, input := range group {
                buf := new(bytes.Buffer)
                json.NewEncoder(buf).Encode(input)
                req, _ := http.NewRequest("POST", g.nodes[idx]+"/api/predict", buf)
                req.Header.Set("Content-Type", "application/json")
                client := &http.Client{}
                resp, err := client.Do(req)
                if err != nil {
                    errs[idx] = err
                    return
                }
                defer resp.Body.Close()
                var result interface{}
                json.NewDecoder(resp.Body).Decode(&result)
                pos := (j*numNodes)+idx
                results[pos] = result
            }
        }(idx, group)
    }
    wg.Wait()

    for _, err := range errs {
        if err != nil {
            http.Error(w, "Error llamando a un nodo: "+err.Error(), http.StatusBadGateway)
            return
        }
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(results)
}

