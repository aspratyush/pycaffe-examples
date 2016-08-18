#!/bin/bash

data=`cat solve.log | grep ", loss = " | awk '{ print $6,$9 }'`
echo "$data" > solve1.log
