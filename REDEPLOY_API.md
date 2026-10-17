# Post-merge Redeploy API

The VPS runs a small HTTP API (in the `vps-infra` repo, `post-merge-api/`) that can redeploy this whole project on demand. Sending it a request pulls the latest code from this repo and recreates every container in [`docker-compose.yml`](docker-compose.yml): `api`, `ui`, `prometheus`, `loki`, `promtail`, and `grafana`.

## Endpoint

```
POST http://<vps-host>:3003/post-merge
```

Port `3003` is the host port that maps to the API container. Put it behind Nginx Proxy Manager with HTTPS if it is reachable from the internet.

### Authentication

HTTP Basic auth. The username and password are in `post-merge-api/.env` in the `vps-infra` repo. Requests without valid credentials get `401 Unauthorized`.

### Request

```bash
curl -X POST http://<vps-host>:3003/post-merge \
  -u "<API_USERNAME>:<API_PASSWORD>" \
  -H "Content-Type: application/json" \
  -d '{"action": "MLOps-Redeploy-All"}'
```

The body must be a JSON object with `"action": "MLOps-Redeploy-All"`.

### Responses

| Status | Meaning |
| --- | --- |
| `202 Accepted` | Redeploy started in the background. The response returns before the deploy finishes. |
| `400 Bad Request` | Body is not a JSON object, or `action` is not `MLOps-Redeploy-All`. |
| `401 Unauthorized` | Missing or wrong Basic auth credentials. |
| `409 Conflict` | A redeploy is already running. |

## What the redeploy does

[`post-merge-api/redeploy-mlops.sh`](https://github.com/rising-entropy/vps-infra/blob/main/post-merge-api/redeploy-mlops.sh) runs these steps in `/root/projects/MLOps-Assignment-1`:

1. `git pull --ff-only` to fetch the latest commit on the checked-out branch. If the local branch has diverged, the pull fails and nothing is redeployed.
2. `docker compose up -d --build --force-recreate --remove-orphans` to rebuild the images and recreate all containers in this project.

Output is logged by the `post-merge-api` container. View it with:

```bash
docker logs post-merge-api
```

## Notes

- The API container mounts the Docker socket and this repo at the same host path. This lets `docker compose` resolve the relative bind mounts (such as `./monitoring/prometheus.yml`) correctly.
- Only one redeploy runs at a time. Requests that arrive during a deploy get `409`.
- Changes to the `vps-infra` side (the API itself) are not picked up by this endpoint. Rebuild that stack by hand.
