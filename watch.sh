#!/usr/bin/env bash

find fragments diagrams media Makefile thesis.md | entr -c make
