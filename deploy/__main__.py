# File based on: https://github.com/pulumi/templates/blob/master/container-gcp-python/__main__.py

import pulumi
import pulumi_docker_build as docker_build
import pulumi_random as random
from pulumi_gcp import (
    artifactregistry,
    compute,
    projects,
    secretmanager,
    serviceaccount,
)

# Consts

# We will use this tag to identify the webserver container and whitelist it in the firewall.
WEBSERVER_TAG = "webserver"
APP_PATH = "../"
MILVUS_TOKEN_SECRET_NAME = "milvus_token_secret"

# Import the program's configuration settings.
app_config = pulumi.Config()
image_name = app_config.get("imageName", "webservice")
container_port = app_config.get_int("containerPort", 8000)
cpu = app_config.get_int("cpu", 1)
memory = app_config.get("memory", "1Gi")
concurrency = app_config.get_int("concurrency", 10)

# Web service configuration
milvusdb_uri = app_config.require("milvus_uri")

# Import the provider's configuration settings.
gcp_config = pulumi.Config("gcp")
location = gcp_config.require("region")
project = gcp_config.require("project")

# Create a unique Artifact Registry repository ID
unique_string = random.RandomString(
    "unique-string",
    length=4,
    lower=True,
    upper=False,
    numeric=True,
    special=False,
)
repo_id = pulumi.Output.concat("rag-solution-container-repo-", unique_string.result)

# Create an Artifact Registry repository
repository = artifactregistry.Repository(
    "repository",
    description="Repository for container image",
    format="DOCKER",
    location=location,
    repository_id=repo_id,
)

# Form the repository URL
repo_url = pulumi.Output.concat(
    location, "-docker.pkg.dev/", project, "/", repository.repository_id
)

# Create a container image for the service.
# Before running `pulumi up`, configure Docker for Artifact Registry authentication
# as described here: https://cloud.google.com/artifact-registry/docs/docker/authentication
image = docker_build.Image(
    "image",
    tags=[pulumi.Output.concat(repo_url, "/", image_name)],
    context=docker_build.BuildContextArgs(
        location=APP_PATH,
    ),
    # Cloud Run currently requires x86_64 images
    # https://cloud.google.com/run/docs/container-contract#languages
    platforms=[docker_build.Platform.LINUX_AMD64],
    push=True,
)

compute_network = compute.Network(
    "network",
    auto_create_subnetworks=True,
)

compute_firewall = compute.Firewall(
    "firewall",
    network=compute_network.self_link,
    allows=[
        compute.FirewallAllowArgs(
            protocol="tcp",
            ports=["80"],
        )
    ],
    target_tags=[WEBSERVER_TAG],
    source_ranges=["0.0.0.0/0"],
)

# These are needed so COS can write metrics and logs to Cloud Monitoring and Logging
# Create a Service Account
service_account = serviceaccount.Account(
    "default-vm-sa",
    account_id="default-vm-sa",
    display_name="Default VM Service Account",
)

# Assign roles to the Service Account
metric_writer_role = projects.IAMMember(
    "metric-writer-role",
    role="roles/monitoring.metricWriter",
    member=service_account.email.apply(lambda email: f"serviceAccount:{email}"),
    project=project,
)

logs_writer_role = projects.IAMMember(
    "logs-writer-role",
    role="roles/logging.logWriter",
    member=service_account.email.apply(lambda email: f"serviceAccount:{email}"),
    project=project,
)

telemetry_writer_role = projects.IAMMember(
    "telemetry-writer-role",
    role="roles/telemetry.metricsWriter",
    member=service_account.email.apply(lambda email: f"serviceAccount:{email}"),
    project=project,
)

# This is needed to the VM can pull the secret from Secret Manager
secret_accessor_role = projects.IAMMember(
    "secret-accessor-role",
    role="roles/secretmanager.secretAccessor",
    member=service_account.email.apply(lambda email: f"serviceAccount:{email}"),
    project=project,
)

# Lookup the MILVUSDB_TOKEN secret to ensure it exists
milvusdb_token_secret = secretmanager.Secret.get(
    "milvusdb-token-secret",
    pulumi.Output.concat("projects/", project, "/secrets/", MILVUS_TOKEN_SECRET_NAME),
)

# A simple bash script that will run the webserver container on the VM.
# MILVUSDB_TOKEN is fetched securely from Secret Manager at runtime.
startup_script = pulumi.Output.all(image.tags[0], container_port, milvusdb_uri).apply(
    lambda args: f"""#!/bin/bash

SECRET_ID="{MILVUS_TOKEN_SECRET_NAME}"
# The version of the secret to access (e.g., "latest" or a version number)
SECRET_VERSION="latest"
MILVUSDB_TOKEN=$(docker run --rm google/cloud-sdk:slim gcloud secrets versions access "$SECRET_VERSION" --secret="$SECRET_ID")

# Check if the secret was fetched successfully
if [ -n "$MILVUSDB_TOKEN" ]; then
    echo "Secret fetched successfully."
else
    echo "ERROR: Failed to fetch secret." >&2
    exit 1
fi

# Pull the latest version of the container image from Artifact Registry
sudo docker pull {args[0]}

# Run docker container from the built image, exposing the specified port on 80
sudo docker run -d --restart=always -p {args[1]}:80 \
  -e MILVUSDB_URI='{args[2]}' \
  -e MILVUSDB_TOKEN="$MILVUSDB_TOKEN" \
  {args[0]}
"""
)

instance_addr = compute.address.Address("address")
compute_instance = compute.Instance(
    "instance",
    # Force redeploy if the startup_script changes
    #  This is needed to ensure the VM runs the latest version of the webserver container.
    replace_on_changes=["metadata_startup_script"],
    machine_type="f1-micro",
    metadata_startup_script=startup_script,
    boot_disk=compute.InstanceBootDiskArgs(
        initialize_params=compute.InstanceBootDiskInitializeParamsArgs(
            image="cos-cloud/cos-117-lts", type="pd-standard"
        )
    ),
    service_account=compute.instance.InstanceServiceAccountArgs(
        email=service_account.email, scopes=["cloud-platform"]
    ),
    network_interfaces=[
        compute.InstanceNetworkInterfaceArgs(
            network=compute_network.id,
            access_configs=[
                compute.InstanceNetworkInterfaceAccessConfigArgs(
                    nat_ip=instance_addr.address
                )
            ],
        )
    ],
    opts=pulumi.ResourceOptions(depends_on=[compute_firewall]),
    tags=[WEBSERVER_TAG],
)

pulumi.export("instanceName", compute_instance.name)
pulumi.export("instanceIP", instance_addr.address)
