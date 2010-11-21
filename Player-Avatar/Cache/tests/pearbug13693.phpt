--TEST--
pearbug13693
--FILE--
<?php

    require_once 'Cache/Lite.php';
    
    // Create temp dir
    $dir = dirname( __FILE__ ) . '/' . uniqid();
    mkdir( $dir );
    
    $options = array(
        'cacheDir' => $dir,
        'lifeTime' => 60,
    );
    $id = '#13693';
    $cache = new Cache_Lite($options);
    $cache->save('stuff', $id);
    // Must be true
    echo ( $cache->remove($id) === true ) ? "OK\n" : "ERROR\n";
    // Will return a PEAR Error
    echo ( $cache->remove($id) instanceof PEAR_Error ) ? "OK\n" : "ERROR\n";
    // Will return true
    echo ( $cache->remove($id, 'default', true) === true ) ? "OK\n" : "ERROR\n";
    
    // Remove temp dir
    rmdir( $dir );
--EXPECT--
OK
OK
OK