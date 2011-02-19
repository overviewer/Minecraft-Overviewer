<?php

// Bench script of Cache_Lite
// $Id: bench.php,v 1.6 2002/09/28 18:05:29 fab Exp $

require_once('Cache/Lite.php');

$options = array(
    'caching' => true,
    'cacheDir' => '/tmp/',
    'lifeTime' => 10
);

$Cache_Lite = new Cache_Lite($options);

if ($data = $Cache_Lite->get('123')) {
    echo($data);
} else {
    $data = '';
    for($i=0;$i<1000;$i++) {
        $data .= '0123456789';
    }
    echo($data);
    $Cache_Lite->save($data);
}

?>