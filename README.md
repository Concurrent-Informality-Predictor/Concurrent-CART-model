# ML Informalidad Laboral - Modelo Concurrente en Go

Este proyecto implementa un modelo de Machine Learning para predecir la informalidad laboral usando programación concurrente y distribuida en Go.

##  Arquitectura del Sistema

El sistema está compuesto por los siguientes componentes:

- **Modelo ML**: CART implementado desde cero en Go
- **Sistema Concurrente**: Uso de goroutines y canales para procesamiento paralelo
- **Cache Distribuido**: Redis para almacenamiento temporal del modelo entrenado
- **API REST**: Servidor HTTP para predicciones en tiempo real

## Integrantes

- Samuel Cano (U202116508@upc.edu.pe) 🐱
- Eduardo Puglisevic (U202115535@upc.edu.pe)
- Nicolás Guerrero (U20201e850@upc.edu.pe)
