<?php

/*
 * This code is taken from PEAR/File from Michael Wallner <mike@php.net>
 * to avoid an extra dependency for a single function.
 *
 * LICENSE: This source file is subject to version 3.0 of the PHP license
 * that is available through the world-wide-web at the following URI:
 * http://www.php.net/license/3_0.txt.  If you did not receive a copy of
 * the PHP License and are unable to obtain it through the web, please
 * send a note to license@php.net so we can mail you a copy immediately.
 */
 
require_once('PEAR.php');
define('FILE_WIN32', defined('OS_WINDOWS') ? OS_WINDOWS : !strncasecmp(PHP_OS, 'win', 3));   

/**
 * Returns the temp directory according to either the TMP, TMPDIR, or
 * TEMP env variables. If these are not set it will also check for the
 * existence of /tmp, %WINDIR%\temp
 *
 * @static
 * @access  public
 * @return  string  The system tmp directory
 */
function tmpDir()
{
    if (FILE_WIN32) {
        if (isset($_ENV['TEMP'])) {
            return $_ENV['TEMP'];
        }
        if (isset($_ENV['TMP'])) {
            return $_ENV['TMP'];
        }
        if (isset($_ENV['windir'])) {
            return $_ENV['windir'] . '\\temp';
        }
        if (isset($_ENV['SystemRoot'])) {
            return $_ENV['SystemRoot'] . '\\temp';
        }
        if (isset($_SERVER['TEMP'])) {
            return $_SERVER['TEMP'];
        }
        if (isset($_SERVER['TMP'])) {
            return $_SERVER['TMP'];
        }
        if (isset($_SERVER['windir'])) {
            return $_SERVER['windir'] . '\\temp';
        }
        if (isset($_SERVER['SystemRoot'])) {
            return $_SERVER['SystemRoot'] . '\\temp';
        }
        return '\temp';
    }
    if (isset($_ENV['TMPDIR'])) {
        return $_ENV['TMPDIR'];
    }
    if (isset($_SERVER['TMPDIR'])) {
        return $_SERVER['TMPDIR'];
    }
    return '/tmp';
}

?>
