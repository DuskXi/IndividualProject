:- modeh(1, true_class(+example)).

%count_objects(Example, Shape, Material, Count) :-
%    findall(Object, (
%        contains(Object, _, Example),
%        has_shape(Object, Shape), %member(Shape, Shapes),
%        has_material(Object, Material) %,member(Material, Materials)
%    ), Objects),
%    length(Objects, Count).

% test_object(A) :-
%     contains(B,A), has_material(B, rubber), left_of(C,B), has_shape(C, cube),
%     print(B), print(C).
% test_object(A) :-
%     contains(B,A), has_material(B, rubber), left_of(C,B), has_shape(C, cylinder),
%     print(B), print(C).

:- modeb(*, contains(-object, +example)).
:- modeb(*, has_shape(+object, #shape)).
:- modeb(*, has_material(+object, #material)).
:- modeb(*, has_color(+object, #color)).
:- modeb(*, has_size(+object, #size)).
:- modeb(*, left_of(+object, -object)).
:- modeb(*, right_of(+object, -object)).
:- modeb(*, front_of(+object, -object)).
:- modeb(*, behind_of(+object, -object)).
% :- modeb(*, count_objects(+example, +shape, +material, #count)).
% :- modeb(*, test_object(+example)).

:- determination(true_class/1, contains/2).
:- determination(true_class/1, has_shape/2).
:- determination(true_class/1, has_material/2).
:- determination(true_class/1, has_color/2).
:- determination(true_class/1, has_size/2).
:- determination(true_class/1, left_of/2).
:- determination(true_class/1, right_of/2).
:- determination(true_class/1, front_of/2).
:- determination(true_class/1, behind_of/2).
% :- determination(true_class/1, test_object/1).
%:- determination(true_class/1, count_objects/4).


:- set(i,4).
:- set(verbosity,2).
:- set(minpos,3).
:- set(noise,0).
:- set(clauselength, 20).
:- consult('example.bk').
