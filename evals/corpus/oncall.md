# On-call handbook

Each engineering team runs a weekly on-call rotation. The rotation changes every Monday at 10:00 local time, and the outgoing engineer writes a short handover note listing open incidents and noisy alerts.

## Paging and acknowledgement

A page must be acknowledged within 5 minutes. If the primary does not acknowledge, the pager escalates to the secondary, and after 15 minutes to the engineering manager.

## Severity levels

SEV1 means customers cannot place orders or log in; it requires an incident commander and a status page update within 30 minutes. SEV2 means a major feature is degraded for many customers. SEV3 is a minor bug or a single-customer issue and can wait for business hours.

## Postmortems

Every SEV1 and SEV2 incident gets a blameless postmortem within five working days. The postmortem lists the timeline, the root cause, what went well, and action items with owners and due dates.

## Compensation

Engineers on call receive a fixed weekly allowance and can take a day off after a night with more than two pages.
