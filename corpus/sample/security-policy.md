# Security Policy

All services must authenticate internal requests with short-lived tokens;
static shared secrets are prohibited in new code. Secrets live in the vault,
never in environment files committed to source control.

## Incident response

Suspected incidents are reported in the #security channel immediately. The
on-call security engineer triages within 30 minutes during business hours and
2 hours otherwise. Severity-1 incidents trigger the paging rotation and an
incident commander is assigned.

## Data handling

Customer PII is stored only in the primary database with row-level security
enabled. Exports containing PII require a signed data-processing ticket and
auto-expire after 7 days.
