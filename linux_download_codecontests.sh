#!/bin/bash

# Create a folder to designate for codecontests data
mkdir -p /tmp/codecontests

# Build the Bazel target
bazel build -c opt :print_names_and_sources

# Copy the codecontests data from the Google Cloud Storage bucket to the designated folder
gsutil -m cp -r gs://dm-code_contests /tmp/codecontests
