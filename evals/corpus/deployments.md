# Deployment process

All services are deployed from the `main` branch through the release pipeline. A merge to `main` builds a container image, tags it with the commit SHA and pushes it to the registry.

## Staging first

Every release goes to the staging environment before production. Staging runs the same container image as production, with anonymised data. The release owner runs the smoke test suite against staging and records the result in the release ticket.

## Production rollout

Production uses a canary rollout. The new version first receives 10 percent of traffic for fifteen minutes. If the error rate and p95 latency stay within the service level objectives, the rollout continues to 50 percent and then 100 percent.

## Rollback

To roll back, redeploy the previous image tag from the release history. Database migrations must be backwards compatible for one release, so the previous image can run against the new schema. Never edit a migration that has already run in production; write a new one instead.

## Deploy freeze

There is no production deploy on Friday after 15:00 or during the last week of December unless the incident commander approves a hotfix.
