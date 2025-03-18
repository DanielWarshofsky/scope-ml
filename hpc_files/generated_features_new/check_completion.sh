#! /bin/bash
total_files=64
for field in field*
do
count=$(ls ${field}/*.parquet -1 | wc -l)
if [ $count -ne $total_files ]
then
echo "${field} does not have 64 feature files it has ${count}"
else
echo "${field} is complete"
fi
done
