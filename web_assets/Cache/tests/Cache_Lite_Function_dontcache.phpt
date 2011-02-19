--TEST--
Cache_Lite::Cache_Lite_Function (dont cache)
--FILE--
<?php

require_once('callcache.inc');
require_once('tmpdir.inc');
require_once('cache_lite_function_base.inc');

// Classical
$options = array(
    'cacheDir' => tmpDir() . '/',
    'lifeTime' => 60,
    'debugCacheLiteFunction' => true
);
$cache = new Cache_Lite_Function($options);
$data = $cache->call('function_test', 23, 66);
echo($data);
$data = $cache->call('function_test', 23, 66);
echo($data);
$cache->clean();

// Don't Cache if output contains NOCACHE
$options = array(
    'cacheDir' => tmpDir() . '/',
    'lifeTime' => 60,
    'debugCacheLiteFunction' => true,
    'dontCacheWhenTheOutputContainsNOCACHE' => true
);
$cache = new Cache_Lite_Function($options);
$data = $cache->call('function_test2', 23, 66);
echo($data);
$data = $cache->call('function_test2', 23, 66);
echo($data);
$data = $cache->call('function_test2', 0, 66);
echo($data);
$data = $cache->call('function_test2', 0, 66);
echo($data);
$cache->clean();

// Don't cache if result if false
$options = array(
    'cacheDir' => tmpDir() . '/',
    'lifeTime' => 60,
    'debugCacheLiteFunction' => true,
    'dontCacheWhenTheResultIsFalse' => true
);
$cache = new Cache_Lite_Function($options);
$data = $cache->call('function_test', 23, 66);
echo($data);
$data = $cache->call('function_test', 23, 66);
echo($data);
$data = $cache->call('function_test', 0, 66);
echo($data);
$data = $cache->call('function_test', 0, 66);
echo($data);
$cache->clean();

// Don't cache if result if null
$options = array(
    'cacheDir' => tmpDir() . '/',
    'lifeTime' => 60,
    'debugCacheLiteFunction' => true,
    'dontCacheWhenTheResultIsNull' => true
);
$cache = new Cache_Lite_Function($options);
$data = $cache->call('function_test', 23, 66);
echo($data);
$data = $cache->call('function_test', 23, 66);
echo($data);
$data = $cache->call('function_test', 1, 66);
echo($data);
$data = $cache->call('function_test', 1, 66);
echo($data);
$cache->clean();

function function_test($arg1, $arg2) 
{
    echo "This is the output of the function function_test($arg1, $arg2) !\n";
    if ($arg1==0) {
        return false;
    }
    if ($arg1==1) {
        return null;
    }
    return '';
}

function function_test2($arg1, $arg2) 
{
    if ($arg1==0) {
        echo "NOCACHE";
    }
    return "This is the result of the function function_test2($arg1, $arg2) !\n";
}

?>
--GET--
--POST--
--EXPECT--
Cache missed !
This is the output of the function function_test(23, 66) !
Cache hit !
This is the output of the function function_test(23, 66) !
Cache missed !
This is the result of the function function_test2(23, 66) !
Cache hit !
This is the result of the function function_test2(23, 66) !
Cache missed !
This is the result of the function function_test2(0, 66) !
Cache missed !
This is the result of the function function_test2(0, 66) !
Cache missed !
This is the output of the function function_test(23, 66) !
Cache hit !
This is the output of the function function_test(23, 66) !
Cache missed !
This is the output of the function function_test(0, 66) !
Cache missed !
This is the output of the function function_test(0, 66) !
Cache missed !
This is the output of the function function_test(23, 66) !
Cache hit !
This is the output of the function function_test(23, 66) !
Cache missed !
This is the output of the function function_test(1, 66) !
Cache missed !
This is the output of the function function_test(1, 66) !
