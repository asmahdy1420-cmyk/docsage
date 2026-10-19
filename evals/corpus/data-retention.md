# Data retention

This policy defines how long we keep each category of data and how it is deleted.

## Retention periods

Application access logs are kept for 90 days and then deleted automatically. Audit logs for administrative actions are kept for seven years because of financial regulations. Customer support conversations are kept for two years after the ticket is closed.

## Backups

Database backups are taken every night and kept for 35 days. Backups are encrypted with a key held in the cloud key management service, and a restore test is performed every quarter.

## Account deletion

When a customer deletes their account, personal data is removed from the primary database within 30 days. Invoices are kept, with the name and address preserved only as legally required, because tax law requires them.

## Personal data in analytics

The analytics warehouse stores pseudonymous identifiers, not email addresses or phone numbers. Raw event data older than 13 months is aggregated and the raw rows are dropped.
