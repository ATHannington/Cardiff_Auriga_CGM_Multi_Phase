C FILE: where_duplicates.F90
      SUBROUTINE where_duplicates(A,B,N,M,OUT,OUT_INDICES)
C
C     Find common entries bewteen A and B including all duplicates
C     Returns: OUT : Matching Values of A in B, -1 where no match
C              OUT_INDICES: B Indices of Matching Values of A in B in Python
C                           0 start numbering
C
          IMPLICIT NONE
          INTEGER :: N,M
          REAL*8, DIMENSION(N)                   :: A
          REAL*8, DIMENSION(N)                :: OUT
          REAL*8, DIMENSION(M)                   :: B
          INTEGER, DIMENSION(N)               :: OUT_INDICES
          INTEGER               :: ii, jj
Cf2py     intent(in)            :: A,B
Cf2py     intent(out)           :: OUT, OUT_INDICES


            OUT = -1.d0
            OUT_INDICES = -1
            
            DO ii=1,SIZE(B)
              DO jj=1, SIZE(A)
                IF (A(jj) .EQ. B(ii)) THEN
                  OUT(jj) = B(ii)
                  OUT_INDICES(jj) = ii-1
                ENDIF
              END DO
            END DO

      END SUBROUTINE where_duplicates
C END FILE where_duplicates.F90
