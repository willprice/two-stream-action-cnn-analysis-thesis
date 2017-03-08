#!/usr/bin/env bash

find fragments diagrams Makefile thesis.md | entr -c make
