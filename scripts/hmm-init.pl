#!/usr/bin/perl

use strict;
use warnings;
use File::Basename;
use Pod::Usage;
use Getopt::Long;

my %opt = (man => 0, help => 0, hmmmacro => "hmmmacro.mmf");
GetOptions(\%opt,'help|?|h','man|m', 'hmmmacro|o=s', 'infofile|i=s'
, 'variances|v=s', 'covariances|c=s', 'means|e=s') or pod2usage(2);
pod2usage(1) if $opt{help};
pod2usage(-exitstatus => 0, -verbose => 2) if $opt{man};


my $lshort = length( pack("s",1) );
my $lint = length( pack("i",1) );
my $lfloat = length( pack("f",1) );
my $N = -1;
my $DIM = -1;
my @means = ();
my @cov = ();

# calculating the means, variances and covariances
while(<>) {
    chomp;
    open FILE, $_;
    print "$_\n";
    binmode FILE;
    my ($val, $num, $dummy1, $dummy2);
    read( FILE, $num, $lint ); $num = unpack( "i", $num );
    read( FILE, $dummy1, $lint ); $dummy1 = unpack( "i", $dummy1 );
    read( FILE, $DIM, $lshort ); $DIM = unpack( "s", $DIM ); $DIM /= $lfloat;
    read( FILE, $dummy2, $lshort ); $dummy2 = unpack( "s", $dummy2 );
    if ( $N == -1 )  {  #initialize
        for( my $j = 0; $j < $DIM; $j++ ) {
            $means[$j] = 0;
            for( my $k = $j; $k < $DIM; $k++ ) {
                $cov[$j][$k] = 0;
            }
        }
        $N = 0;
    }
    close FILE;
}

my @variances;
for(my $i=0;$i<$DIM;$i++) {
    push @variances, $cov[$i][$i];
}

# initializing the hmms
my @names = ();
my @states = ();
my @cov_info = ();
my @init = ();
my $cov_info = 0;
open INFO, "$opt{infofile}" or die "could not read from $opt{infofile}\n";
while(<INFO> )
{
    /(\w+)\s+(\d+)\s+(cov|nocov)\s+(init|noinit)/;
    push @names, $1;
    push @states, $2;
    push @cov_info, $3;
    $cov_info = 1 if ( $3 eq "cov" );
    push @init, $4;
}
close INFO;

open MMF, ">$opt{hmmmacro}" or die "could not write to $opt{hmmmacro}\n";
select MMF;
my $vecsize = @means;
print "~o\n<STREAMINFO> 1 $vecsize\n<VECSIZE> $vecsize<NULLD><USER>";
print "<DIAGC>" if $cov_info == 0;
print "\n";

for(my $i = 0; $i < @names; $i++ ) {
  print  "~h \"$names[$i]\"\n";
  print  "<BEGINHMM>\n";
  print  "<NUMSTATES> $states[$i]\n";
  for( my $j = 2; $j < $states[$i]; $j++ )   {
    print  "<STATE> $j\n";
    print  "<MEAN> $vecsize\n";
    for(my $k=0; $k<@means; $k++ ) {
      if ( $init[$i] eq "noinit" ) {
        print  " ".( 1.0 / $vecsize );
      }
      else {
        print  " $means[$k]";
      }
    }
    print  "\n";
    if ( $cov_info[$i] eq "nocov" )  {
      print  "<VARIANCE> $vecsize\n";
      for(my $k=0; $k < @variances; $k++ )   {
        if ( $init[$i] eq "noinit" )  {
          print  " ".( 1.0 / $vecsize );
        }
        else {
          print  " $variances[$k]";
        }
      }
      print  "\n";
    }
    else {
      print  "<INVCOVAR> $vecsize\n";
      for(my $k = 0; $k < @variances; $k++ )      {
        if ( $init[$i] eq "noinit" )   {
          print  " ".( 1.0 / $vecsize );
        }
        else  {
          print  " ".( 1.0 / $variances[$k] );
        }
        for( my $l = $k + 1; $l < @variances; $l++ )  {
          print  " 0.0";
        }
        print  "\n";
      }
    }
  }
  print  "<TRANSP> $states[$i]\n";
  for( my $j = 1; $j <= $states[$i]; $j++ )  {
    for(my $k = 1; $k <= $states[$i]; $k++ )    {
      if ( $j == 1 && $k == 2 )     {
        print  " 1.0";
      }
      elsif ( $j > 1 && $j < $states[$i] && ( $k == $j || $k == $ j + 1 ) )       {
        print  " 0.5";
      }
      else      {
        print  " 0.0";
      }
    }
    print  "\n";
  }
  print  "<ENDHMM>\n";
}
close MMF;


__END__

=head1 NAME

inithmm - initialize the HMM macro file.

=head1 SYNOPSIS

inithmm [-mh] [-o hmmmacro] -i infofile trainlist

=head1 DESCRPIPTION

B<inithmm> extracts information (means, variances, covariances) of the feature vector file (*.htk) listes in I<trainlist> and creates
a new HMM macro file which is written in I<hmmmacro>.

=head1 OPTIONS

=over 8

=item B<-h, -help>

Print a brief help message and exits.

=item B<-m, --man>

Prints the manual page and exits.

=item B<-o, --hmmmacro>

Specify the HMM macro file, which will contain the result (default: hmmmacro.mmf)

=item B<-i, --infofile=file>

Specify the HMM info file, which contains the information about the character HMMs
A example line looks like: A 20 nocov init

=item B<-e, --means=file>

Print the calculates means to I<file>.

=item B<-v, --variances=file>

Print the calculates variances to I<file>.

=item B<-c, --covariances=file>

Print the calculates covariances to I<file>.

=back

=head1 AUTHOR

Roman Bertolami 2004, bertolam@iam.unibe.ch

=head1 CONTRIBUTOR

Markus Wüthrich 2007-2008, kusi.w@students.unibe.ch

=cut
