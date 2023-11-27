 <?php
 session_start();
error_reporting(0);
include('includes/config.php');

$uid=$_SESSION['id'];
$actionrec = $_GET['status'];
$pid = $_GET['pid'];

$ret = mysqli_query($con,"SELECT * FROM recommended WHERE productId='$pid' and userId='".$_SESSION['id']."'");
$num=mysqli_num_rows($ret);
if($num>0)
{
	header('location:order-history.php');
}else {

mysqli_query($con,"insert into recommended(userId,productId,status) values('".$_SESSION['id']."','$pid','$actionrec')");
header('location:order-history.php');
}
?>
 
 
 