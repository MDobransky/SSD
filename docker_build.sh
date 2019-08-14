#!/bin/bash
docker build -t pytorch:v1.0 -f ./docker/pytorch .
docker build -t ssd:v1.0 -f ./docker/SSD .
