#!/bin/bash
# code obtained from : https://aaronsplace.co.uk/blog/2015-12-22-caffe-log-plotting.html

data=`cat solve.log | grep ", loss = " | awk '{ print $6,$9 }'`
echo "$data" > solve1.log
