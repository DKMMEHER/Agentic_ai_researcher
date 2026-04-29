# GCP Deployment CI/CD Pipeline

This plan outlines the steps to build and automatically deploy the multi-agent researcher to Google Cloud Platform (specifically Cloud Run) using GitHub Actions.

## User Review Required

> [!IMPORTANT]
> The automated deployment requires a Google Cloud Project with Billing enabled, Artifact Registry set up, and Cloud Run API enabled. You will also need to configure specific secrets in your GitHub repository.

## Proposed Changes

### Setup Deployment Workflow

I will create a new workflow file specifically for CD that runs *after* your existing `CI` workflow completes successfully on the `main` branch.

#### [NEW] `.github/workflows/cd.yml`
This workflow will:
1. Wait for the `CI` workflow (linting and testing) to succeed using the `workflow_run` trigger.
2. Authenticate to Google Cloud using Workload Identity Federation (recommended for security) or Service Account Credentials.
3. Build the Docker image using the existing `Dockerfile`.
4. Tag and push the image to Google Artifact Registry (GAR).
5. Deploy the container to Google Cloud Run, passing in the necessary environment variables securely.

## Open Questions

> [!WARNING]
> Before I proceed, please answer the following questions so I can correctly configure the deployment workflow:

1. **Authentication Policy**: Do you prefer to use Google Workload Identity Federation (more secure, no long-lived keys) or a traditional Service Account JSON Key stored as a GitHub Secret?
2. **GCP Project ID**: What is the ID of your Google Cloud Project? here is the GCP project ID: **agentic-ai-researcher**

3. **Target Region**: To which Google Cloud region do you want to deploy? (e.g., `us-central1`, `europe-west4`) keep target reioon as asia-south1 (Mumbai)
4. **Environment Variables**: The application requires several keys (e.g., `GROQ_API_KEY`, `TAVILY_API_KEY`, `GOOGLE_API_KEY`). Should they be pulled from Google Cloud Secret Manager at runtime, or injected directly as Environment Variables during the Cloud Run deployment via GitHub Secrets?

## Verification Plan

### Manual Verification
1. I will write the `.github/workflows/cd.yml` file.
2. You will configure the necessary GCP resources (Artifact Registry, Service Account) and GitHub Secrets.
3. You will commit and push the changes.
4. We will monitor the GitHub Actions panel to ensure the build, push, and deploy steps complete successfully and the Cloud Run service URL is active.
