--TEST--
Cache_Lite::Cache_Lite_Function (classical)
--FILE--
<?php

require_once('callcache.inc');
require_once('tmpdir.inc');
require_once('cache_lite_function_base.inc');

$options = array(
    'cacheDir' => tmpDir() . '/',
    'lifeTime' => 60
);

$cache = new Cache_Lite_Function($options);

$data = $cache->call('function_to_bench', 23, 66);
echo($data);
$data = $cache->call('function_to_bench', 23, 66);
echo($data);
$cache->call('function_to_bench', 23, 66);

$object = new bench();
$object->test = 666;
$data = $cache->call('object->method_to_bench', 23, 66);
echo($data);
$data = $cache->call('object->method_to_bench', 23, 66);
echo($data);
$cache->call('object->method_to_bench', 23, 66);

$data = $cache->call('bench::static_method_to_bench', 23, 66);
echo($data);
$data = $cache->call('bench::static_method_to_bench', 23, 66);
echo($data);
$cache->call('bench::static_method_to_bench', 23, 66);

$object = new test($options);

$cache->clean();

function function_to_bench($arg1, $arg2) 
{
    echo "This is the output of the function function_to_bench($arg1, $arg2) !\n";
    return "This is the result of the function function_to_bench($arg1, $arg2) !\n";
}

class bench
{
    var $test;

    function method_to_bench($arg1, $arg2)
    {
        echo "\$obj->test = $this->test and this is the output of the method \$obj->method_to_bench($arg1, $arg2) !\n";
        return "\$obj->test = $this->test and this is the result of the method \$obj->method_to_bench($arg1, $arg2) !\n";        
    }
    
    function static_method_to_bench($arg1, $arg2) {
        echo "This is the output of the function static_method_to_bench($arg1, $arg2) !\n";
        return "This is the result of the function static_method_to_bench($arg1, $arg2) !\n";
    }

}

class test
{
    function test($options) {
        $this->foo = 'bar';
        $cache = new Cache_Lite_Function($options);
        echo($cache->call(array($this, 'method_to_bench'), 'foo', 'bar'));
    }   
    
    function method_to_bench($arg1, $arg2)
    {
        echo "output : *** $arg1 *** $arg2 *** " . $this->foo . " ***\n";
        return "result : *** $arg1 *** $arg2 *** " . $this->foo . " ***\n";     
    }
}

?>
--GET--
--POST--
--EXPECT--
This is the output of the function function_to_bench(23, 66) !
This is the result of the function function_to_bench(23, 66) !
This is the output of the function function_to_bench(23, 66) !
This is the result of the function function_to_bench(23, 66) !
This is the output of the function function_to_bench(23, 66) !
$obj->test = 666 and this is the output of the method $obj->method_to_bench(23, 66) !
$obj->test = 666 and this is the result of the method $obj->method_to_bench(23, 66) !
$obj->test = 666 and this is the output of the method $obj->method_to_bench(23, 66) !
$obj->test = 666 and this is the result of the method $obj->method_to_bench(23, 66) !
$obj->test = 666 and this is the output of the method $obj->method_to_bench(23, 66) !
This is the output of the function static_method_to_bench(23, 66) !
This is the result of the function static_method_to_bench(23, 66) !
This is the output of the function static_method_to_bench(23, 66) !
This is the result of the function static_method_to_bench(23, 66) !
This is the output of the function static_method_to_bench(23, 66) !
output : *** foo *** bar *** bar ***
result : *** foo *** bar *** bar ***
