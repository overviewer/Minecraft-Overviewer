<?php

function callCache($id, $type = 'string') {
    global $Cache_Lite;
    if ($data = $Cache_Lite->get($id)) {
        echo("Cache Hit !\n");
        if ($type=='string') {
            echo($data);
        }
        if ($type=='array') {
            echo(serialize($data));
        }
    } else {
        echo("Cache Missed !\n");
        if ($type=='string') {
            $data = '';
            for($i=0;$i<10;$i++) {
                $data .= '0123456789';
            }
            echo($data);
        }
        if ($type=='array') {
            $data = array(array('foo', 'bar'), 1, 'foo', 'bar');
            echo(serialize($data));
        }
        $res = $Cache_Lite->save($data);
        if (is_object($res)) {
            echo "\nPEAR_ERROR : " . $res->getMessage() . " (#" . $res->getCode() . ")\n";
        } else {
	        if (!($res)) {
	            echo "\nError when saving cache !\n";
	        }
        }
    }
}
    
function multipleCallCache($type = 'string') {
    global $Cache_Lite;
    
    echo "==> First call (cache should be missed)\n";
    callCache('31415926', $type);
    echo "\nDone !\n\n";
    
    echo "==> Second call (cache should be hit)\n";
    callCache('31415926', $type);
    echo "\nDone !\n\n";
    
    echo "==> Third call (cache should be hit)\n";
    callCache('31415926', $type);
    echo "\nDone !\n\n";
    
    echo "==> We remove cache\n";
    $Cache_Lite->remove('31415926');
    echo "Done !\n\n";
    
    echo "==> Fourth call (cache should be missed)\n";
    callCache('31415926', $type);
    echo "\nDone !\n\n";
    
    echo "==> #5 Call with another id (cache should be missed)\n";
    callCache('3141592653', $type);
    echo "\nDone !\n\n";
    
    echo "==> We remove cache\n";
    $Cache_Lite->remove('31415926');
    $Cache_Lite->remove('3141592653');
    echo "Done !\n";
}

function callCache2($id, $type = 'string') {
    global $Cache_Lite_Output;
    if (!($Cache_Lite_Output->start($id))) {
        if ($type=='string') {
            $data = '';
            for($i=0;$i<10;$i++) {
                $data .= '0123456789';
            }
            echo($data);
        }
        if ($type=='array') {
            $data = array(array('foo', 'bar'), 1, 'foo', 'bar');
            echo(serialize($data));
        }
        $Cache_Lite_Output->end();
        echo("Cache Missed !\n");
    } else {
        echo("Cache Hit !\n");
    }
}

function multipleCallCache2($type = 'string') {
    global $Cache_Lite_Output;
    
    echo "==> First call (cache should be missed)\n";
    callCache2('31415926', $type);
    echo "\nDone !\n\n";
    
    echo "==> Second call (cache should be hit)\n";
    callCache2('31415926', $type);
    echo "\nDone !\n\n";
    
    echo "==> Third call (cache should be hit)\n";
    callCache2('31415926', $type);
    echo "\nDone !\n\n";
    
    echo "==> We remove cache\n";
    $Cache_Lite_Output->remove('31415926');
    echo "Done !\n\n";
    
    echo "==> Fourth call (cache should be missed)\n";
    callCache2('31415926', $type);
    echo "\nDone !\n\n";
    
    echo "==> #5 Call with another id (cache should be missed)\n";
    callCache2('3141592653', $type);
    echo "\nDone !\n\n";
    
    echo "==> We remove cache\n";
    $Cache_Lite_Output->remove('31415926');
    $Cache_Lite_Output->remove('3141592653');
    echo "Done !\n";
}

function multipleCallCache3_1($type = 'string') {
    global $Cache_Lite;
       
    echo "==> #6 call (cache should be missed)\n";
    callCache('31415926', $type);
    echo "\nDone !\n\n";
    
    echo "==> #7 call (cache should be hit)\n";
    callCache('31415926', $type);
    echo "\nDone !\n\n";
}

function multipleCallCache3_2($type = 'string') {
    global $Cache_Lite;
      
    echo "==> #8 call (cache should be missed)\n";
    callCache('31415926', $type);
    echo "\nDone !\n\n";
    
    echo "==> We remove cache\n";
    $Cache_Lite->remove('31415926');
    echo "Done !\n";
    
}

?>
