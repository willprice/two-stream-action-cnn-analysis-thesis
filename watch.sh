#!/usr/bin/env bash

find fragments diagrams media Makefile thesis.{md,tex} | grep -v clippings | entr -c make
