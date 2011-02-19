--TEST--
Cache_Lite::Cache_Lite_Function (drop() method)
--FILE--
<?php

require_once('callcache.inc');
require_once('tmpdir.inc');
require_once('cache_lite_function_base.inc');

$options = array(
    'cacheDir' => tmpDir() . '/',
    'lifeTime' => 60
);
$foo = 10;

$cache = new Cache_Lite_Function($options);

$data = $cache->call('function_to_bench', 23, 66);
echo($data."\n");
$data = $cache->call('function_to_bench', 23, 66);
echo($data."\n");
$cache->drop('function_to_bench', 23, 66);
$data = $cache->call('function_to_bench', 23, 66);
echo($data."\n");
$data = $cache->call('function_to_bench', 23, 66);
echo($data."\n");
$cache->clean();

function function_to_bench($arg1, $arg2) 
{
    global $foo;
    $foo = $foo + 1;
    echo "hello !\n";
    return($foo + $arg1 + $arg2);
}

?>
--GET--
--POST--
--EXPECT--
hello !
100
hello !
100
hello !
101
hello !
101
