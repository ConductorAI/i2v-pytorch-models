# pytorch image to vector (for Weaviate)
The inference container for the img2vec module

## Documentation

Documentation for this module can be found [here](https://weaviate.io/developers/weaviate/current/retriever-vectorizer-modules/img2vec-neural.html).

## Build Docker container

```
LOCAL_REPO="img2vec-pytorch" MODEL_NAME="resnet50" ./cicd/build.sh
```

## Build with distroless base
Swap `linux/amd64` with `linux/arm64` for arm-based builds
```
docker buildx build --platform linux/amd64 -t [url]/image-2-vector:[tag] -f Dockerfile.distroless --push .
```
