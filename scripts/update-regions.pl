#!/usr/bin/perl
use strict;
use POSIX;


my $map="/minecraft/maps";

my $str="var regionData=\[\n";
open(IN, "/minecraft/world/protectedCuboids.txt");
foreach(<IN>) {
	my ($x1, $y1, $z1, $x2, $y2, $z2, $owners, $name) = m/(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),\s+(.*?),(.*)/;
	# print "$x1, $y1, $z1, $x2, $y2, $z2, $owners, $name\n";
	$str.='	{"color": "#FFAA00", "opacity": 0.5, "closed": true, "path": ['."\n";
	$str.="		{\"x\": $x1, \"y\": $y1, \"z\": $z1},\n";
	$str.="		{\"x\": $x1, \"y\": $y1, \"z\": $z2},\n";
	$str.="		{\"x\": $x2, \"y\": $y1, \"z\": $z2},\n";
	$str.="		{\"x\": $x2, \"y\": $y1, \"z\": $z1},\n";
	chomp($str);
	chomp($str);
	$str.="\n	\]\},\n";
	$str.='	{"color": "#FFAA00", "opacity": 0.5, "closed": true, "path": ['."\n";
	$str.="		{\"x\": $x1, \"y\": $y2, \"z\": $z1},\n";
	$str.="		{\"x\": $x1, \"y\": $y2, \"z\": $z2},\n";
	$str.="		{\"x\": $x2, \"y\": $y2, \"z\": $z2},\n";
	$str.="		{\"x\": $x2, \"y\": $y2, \"z\": $z1},\n";
	chomp($str);
	chomp($str);
	$str.="\n	\]\},\n";
	
	$str.='	{"color": "#FFAA00", "opacity": 0.5, "closed": true, "path": ['."\n";
	$str.="		{\"x\": $x1, \"y\": $y1, \"z\": $z1},\n";
	$str.="		{\"x\": $x1, \"y\": $y2, \"z\": $z1},\n";
	$str.="		{\"x\": $x2, \"y\": $y2, \"z\": $z1},\n";
	$str.="		{\"x\": $x2, \"y\": $y1, \"z\": $z1},\n";
	chomp($str);
	chomp($str);
	$str.="\n	\]\},\n";
	$str.='	{"color": "#FFAA00", "opacity": 0.5, "closed": true, "path": ['."\n";
	$str.="		{\"x\": $x1, \"y\": $y1, \"z\": $z2},\n";
	$str.="		{\"x\": $x1, \"y\": $y2, \"z\": $z2},\n";
	$str.="		{\"x\": $x2, \"y\": $y2, \"z\": $z2},\n";
	$str.="		{\"x\": $x2, \"y\": $y1, \"z\": $z2},\n";
	chomp($str);
	chomp($str);
	$str.="\n	\]\},\n";
	
	$str.='	{"color": "#FFAA00", "opacity": 0.5, "closed": true, "path": ['."\n";
	$str.="		{\"x\": $x1, \"y\": $y1, \"z\": $z1},\n";
	$str.="		{\"x\": $x1, \"y\": $y2, \"z\": $z1},\n";
	$str.="		{\"x\": $x1, \"y\": $y2, \"z\": $z2},\n";
	$str.="		{\"x\": $x1, \"y\": $y1, \"z\": $z2},\n";
	chomp($str);
	chomp($str);
	$str.="\n	\]\},\n";
	$str.='	{"color": "#FFAA00", "opacity": 0.5, "closed": true, "path": ['."\n";
	$str.="		{\"x\": $x2, \"y\": $y1, \"z\": $z1},\n";
	$str.="		{\"x\": $x2, \"y\": $y2, \"z\": $z1},\n";
	$str.="		{\"x\": $x2, \"y\": $y2, \"z\": $z2},\n";
	$str.="		{\"x\": $x2, \"y\": $y1, \"z\": $z2},\n";
	chomp($str);
	chomp($str);
	$str.="\n	\]\},\n";
	#-4110,49,262,-4100,54,254, adasyd,06

}

close(IN);

open(OUT, ">$map/regions.js");
print OUT "$str\n\];\n";
close(OUT);
