FROM golang:1.23

WORKDIR /app
COPY . .
RUN go mod download
RUN go build -o app

COPY consolidate_clean_data.csv .

EXPOSE 8081 8082 8083
CMD ["./app"]