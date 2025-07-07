FROM golang:1.23 AS build

WORKDIR /app
COPY . .
RUN go mod download
RUN go build -o app main.go
RUN go build -o gateway main_gateway.go

FROM golang:1.23 
WORKDIR /app
COPY --from=build /app/app .
COPY --from=build /app/gateway .
COPY --from=build /app/consolidate_clean_data.csv .

EXPOSE 8081 8082 8083 8000