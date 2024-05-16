#!/usr/bin/env bash

docker build --network=host -t test -f packaging/Dockerfile .
