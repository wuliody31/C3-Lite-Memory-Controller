# Security Policy

## Secrets

Never commit:

- GitHub personal access tokens;
- Neo4j passwords;
- Hugging Face tokens;
- private keys;
- `.env` files containing credentials.

Use environment variables or secret-management infrastructure.

## Research Environment

This repository contains research-oriented code and has not undergone a formal production security audit.

## Supported Security Scope

Current security work focuses on:

- secret separation;
- repository hygiene;
- dependency isolation;
- explicit configuration.

The project does not claim production security certification.

## Reporting

If this repository is used collaboratively, report security-sensitive issues privately to the repository owner rather than publishing credentials or exploit details in a public issue.
