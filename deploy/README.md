# RAG-Solution deployment to GCP with Pulumi

This project demonstrates a possible infrastructure deployment for RAG-Solution webservice. 

**DISCLAIMER**: Altough I've done every reasonable effort to verify this project, it had been **not tested**. It serves as a demo of techniques that could be used for deploying such a service.

## Setup

Install Pulumi for you platform. E.g.: Use Brew on macos.
```sh
brew install pulumi/tap/pulumi
```

Use UV in repo root to install "iac" dependecy group to be able to run this.

```sh
uv sync --group iac
```

Setup the stack variables in `Pulumi.STACK_NAME.yaml` files.

Run the deployment preview with pulumi
```sh
pulumi preview
```

## What does this project not cover?

* **Vector DB Deployment**: VectorDB deployment is **not included** in this IaC project.
  * If we don't want to manage the vector DB, SaaS offering (even with Bring-Your-Own-Cloud) is available. Using that will yield to lowest operational overhead.
  * If we do want to manage our deployment and use the open-source version: Kubernetes is preferd as this vector db (Milvus) needs secondary services (e.g. ETCD) to be deployed. Also if we are willing to invest that amount of effort, probably the project is so large that we also need to scale it, thus Kubernetes is the right option. For the sake a time I'll not build Helm chart as well for this project, although I'm well capable of it. :)
* **Scalability and high-availability**: In this example horizontal scalability (vertical is achieved on code level, both for CPUs and GPUs) is not taken into consideration. Appropriate cloud services (VM scale set on CPU + load balancer) can be used to achieve that which is made possible by the webserver itself being stateless.
* **Public Access**: Things like SSL termination with Load Balancer, SSL Certificate request and custom domain setup are not taken care of.
* **Data Ingestion Pipeline Deployment**: Data ingestion container can be easily deployed with container runing service on a schedule. E.g.: Google Cloud Run, support scheduled execution.
* **Alerting**: Monitoring is setup for VM and Container but alerting on metrics has not been covered, although it is essential part of operating a project.


# Contents

* **__main__.py**: Contains Pulumi script: VM and container deployment, networking and firewall, secret management, monitoring and logging setup, docker container build a publish (Should be done by a CI process realistically.)
* **Pulumi.STACK_NAME.yaml files**: Configuration for two different stacks for development and productive environments.

# High level description

RAG Solution webservice container is deployed to a VM. Google Container Optimizied OS is used to serve the image. VM Service Account had been setup to be able to push metric and log to Cloud Monitoring. 

Alternative: Google Cloud Run could be used to serve the image. Preferabble we don't expect 24/7 traffic or we expect huge spikes in traffic as Cloud Run can dynamically scale the container, even to zero, thus saving money in case of spikey traffic. Other alternative is Kubernetes which could be desirable for larger projects, but it is considerably more effort to manage and deploy projects to. The positive part is that things like high-availability (zero-downtime upgrades (both infra and application), region failover) and scalability are super simple to achieve with it.

Network had been setup to let through HTTP traffic.

Secret (in case Milvus Token) is pulled from Google Cloud Secret Manager on container startup. **NOTE**: It would be preferable to pass secret to container via file instead of an environment variable. It is more secure, less prone to leaking into logs/etc.

In this scenario IaC code serves as an inpromptu CI as it would build the image and restart the VM, thus triggering redeployment. Image building should be achieved by CI machinery instead of IaC code.

# Configuration options
These can be configured via Pulumi.yamls for each stack
* **gcp:project**: GCP project id.
* **gcp:region**: GCP region
* rag-solution:milvus_uri: Vector DB's url.

# Secrets
* **milvus_token_secret**: The project expect this secret to exists in secret manager. Without it it will fail.
