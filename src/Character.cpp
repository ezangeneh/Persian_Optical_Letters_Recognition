#include "Character.h"


Character::Character()
{
}

bool Character::operator<(const Character &ob) const
{
	return this->bBox.x < ob.bBox.x;

}
Character::~Character()
{
}
