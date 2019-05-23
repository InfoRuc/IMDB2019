# 简介

OLAP数据库常用算法的优化技术与性能对比，主要包括SIMD向量化技术、pthread多线程技术、cache-conscious的分区技术，以及GPU并行计算技术。

## Vector Processing

针对TPC-H的Q1语句的向量化优化技术，其中用到的主要技术是SIMD和基于SIMD四路展开。

## Nest Loop Join

## CPU平台

基于SIMD和分区技术的Nest Loop Join算法。

## GPU平台

主要使用共享内存技术。

## Vector Join

基于SIMD技术、多线程、GPU技术的Vector Join算法，以及这些算法的性能比较。

## CPU平台

使用SIMD技术和多线程并行技术。

## GPU平台

主要使用共享内存技术。



